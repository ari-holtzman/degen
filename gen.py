""" 
GPT2 Generation Script
    Adapted from: https://github.com/huggingface/pytorch-transformers/blob/master/examples/contrib/run_openai_gpt.py
    Itself, adapted from https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/train.py
    It self adapted from https://github.com/openai/finetune-transformer-lm/blob/master/train.py
    Many snippets taken from https://github.com/huggingface/pytorch-transformers/
Gumbel Stochastic Beam Search part adapted from:
    https://github.com/wouterkool/stochastic-beam-search/blob/stochastic-beam-search/fairseq/gumbel.py
"""
import argparse
import os
import logging
import json
from tqdm import tqdm, trange

import numpy as np

import torch
import torch.nn
import torch.nn.functional as F
from torch.distributions.gumbel import Gumbel

from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_dataset(dataset_path, batch_size, device, bs=False):
    """ Loads data from a jsonl file with "tokens" attribute """
    dataset, count, tokens, ends, last_len = [], 0, [], [], None
    with open(dataset_path, encoding='utf_8') as f:
        for line in tqdm(f):
            j = json.loads(line.strip())
            cur_len = len(j['tokens'])
            # beam search batches must only contain contexts of the same length
            if not bs:
                tokens.append(j['tokens'])
                end = cur_len-1
                ends.append(end)
                count += 1
                if count == batch_size:
                    max_len = max(ends)
                    data = torch.zeros(batch_size, max_len+1).long()
                    for b, (toks, end) in enumerate(zip(tokens, ends)):
                        data[b, :end+1] = torch.Tensor(toks)
                    data = data.to(device)
                    dataset.append((data, ends))
                    tokens, ends = [], []
                    count = 0
            else:
                if last_len is None:
                    last_len = cur_len
                elif last_len != cur_len  or count == batch_size:
                    data = torch.zeros(count, last_len).long()
                    for b, (toks, end) in enumerate(zip(tokens, ends)):
                        data[b, :last_len] = torch.Tensor(toks)
                    data = data.to(device)
                    dataset.append((data, ends))
                    tokens, ends = [], []
                    count = 0
                    last_len = cur_len
                tokens.append(j['tokens'])
                ends.append(cur_len-1)
                count += 1
    if bs and len(tokens) > 0:
        data = torch.zeros(count, last_len).long()
        for b, (toks, end) in enumerate(zip(tokens, ends)):
            data[b, :last_len] = torch.Tensor(toks)
        data = data.to(device)
        dataset.append((data, ends))

    return dataset

def decode(model, batch_size, max_len, sep, device, temp=None, k=None, p=None, greedy=None, m=None, init=None):
    """ Main decoding function, beam search is in a separate function """
    if init is None:
        context = torch.full((batch_size, 1), sep, dtype=torch.long, device=device)
    else:
        context, ends = init
        limit = max(ends)
        context = context[:, :limit+1]

    output = [ 
               {
                   'ended'      : False,
                   'tokens'     : [],
                   'len'        : 0,
                   'nll4tok'    : [],
                   'ppl4tok'    : [],
                   'ppl'        : 0
               } 
               
               for _ in range(batch_size)
             ]
    if init is not None:
        for i in range(len(output)):
            output[i]['context'] = context[i].cpu().numpy().tolist()
    
    for i in range(max_len):
        logits = model(context)[0]
        if init is None:
            logits = logits[:, -1, :]
        else:
            nu_logits = torch.zeros(logits.size(0), logits.size(2))
            for b, idx in enumerate(ends):
                nu_logits[b, :] = logits[b, idx, :]
            logits = nu_logits
            del nu_logits
        probs = F.softmax(logits, dim=-1)
        logprobs = F.log_softmax(logits, dim=-1)

        if temp is not None:
            samp_probs = F.softmax(logits.div_(temp), dim=-1)
        else:
            samp_probs = probs.clone()

        if greedy:
            next_probs, next_tokens = probs.topk(1)
            next_logprobs = logprobs.gather(1, next_tokens.view(-1, 1))
        elif k is not None:
            indices_to_remove = samp_probs < torch.topk(samp_probs, k)[0][..., -1, None]
            samp_probs[indices_to_remove] = 0
            if m is not None:
                samp_probs.div_(samp_probs.sum(1).unsqueeze(1))
                samp_probs.mul_(1-m)
                samp_probs.add_(probs.mul(m))
            next_tokens = samp_probs.multinomial(1)
            next_logprobs = samp_probs.gather(1, next_tokens.view(-1, 1)).log()
        elif p is not None: 
            sorted_probs, sorted_indices = torch.sort(samp_probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            sorted_samp_probs = sorted_probs.clone()
            sorted_samp_probs[sorted_indices_to_remove] = 0
            if m is not None:
                sorted_samp_probs.div_(sorted_samp_probs.sum(1).unsqueeze(1))
                sorted_samp_probs.mul_(1-m)
                sorted_samp_probs.add_(sorted_probs.mul(m))
            sorted_next_indices = sorted_samp_probs.multinomial(1).view(-1, 1)
            next_tokens = sorted_indices.gather(1, sorted_next_indices)
            next_logprobs = sorted_samp_probs.gather(1, sorted_next_indices).log()
        else:
            if m is not None:
                samp_probs.div_(samp_probs.sum(1).unsqueeze(1))
                samp_probs.mul_(1-m)
                samp_probs.add_(probs.mul(m))
            next_tokens = samp_probs.multinomial(1)
            next_logprobs = samp_probs.gather(1, next_tokens.view(-1, 1)).log()

        next_cat = next_tokens
        next_tokens, next_logprobs = next_tokens.cpu(), next_logprobs.cpu()
        for b in range(batch_size):
            out = output[b]
            if out['ended']:
                continue
            v = next_tokens[b].item()
            logprob = next_logprobs[b].item()
            out['ended'] = v == sep
            out['tokens'].append(v)
            out['len'] += 1
            out['nll4tok'].append(-logprob)
            out['ppl4tok'].append(np.exp(-logprob))
        if init is None:
            context = torch.cat([context, next_cat], dim=1)
        else:
            filler = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            context = torch.cat([context, filler], dim=1)
            for i, tok in enumerate(next_tokens):
                tok = tok.item()
                ends[i] += 1
                context[i, ends[i]] = tok 

    for b in range(batch_size):
        out = output[b]
        out['ppl'] = np.exp(sum(out['nll4tok']) / out['len'])

    return output

def bs_decode_simplified(model, init, w, max_len, sep, device):
    context, ends = init
    assert (all([ends[i] == ends[0] for i in range(len(ends))]))
    cur_len = ends[0] + 1
    batch_size = context.size(0)
    context_cpu = context.cpu()

    beam = context.repeat(w, 1, 1).transpose(0,1).reshape(batch_size*w, cur_len)
    beam_offset = (torch.arange(batch_size)*w*w).repeat(w, 1).t().reshape(-1).to(device)
    beam_nlls = torch.zeros(batch_size*w, 1, device=device)
    beam_ll = torch.zeros(batch_size, w, device=device)
    best_outputs, best_ll, best_nlls = [[None for _ in range(batch_size)] for _ in range(3)]

    for i in trange(max_len):
        if i == 0:
            logits = model(context)[0][:, -1, :]  # (batch_size, V)
            logprobs = F.log_softmax(logits, -1)  # (batch_size, V)
            w_logprobs, w_tokens = torch.topk(logprobs, w)  # both: (batch_size, w)
            cur_tokens = w_tokens.view(-1)  # (batch_size*w,)
            cur_lls = w_logprobs.view(-1)  # (batch_size*w,)

            beam = torch.cat([beam, cur_tokens.unsqueeze(-1)], -1)  # (batch_size*w, cur_len+1)
            beam_nlls = torch.cat([beam_nlls, cur_lls.unsqueeze(-1)], -1)
            beam_ll += cur_lls.view(batch_size, w)
        else:
            logits = model(beam)[0][:, -1, :]  # (batch_size*w, V)
            logprobs = F.log_softmax(logits, -1)  # (batch_size*w, V)
            w_logprobs, w_tokens = torch.topk(logprobs, w)  # (batch_size*w, w)
            candidate_logprobs = (w_logprobs + beam_ll.view(-1).repeat(w, 1).t()).reshape(batch_size, w*w)
            beam_ll, beam_idxs = candidate_logprobs.topk(w)  # both: (batch_size, w)
            beam_idxs = beam_idxs.view(-1) + beam_offset  # (batch_size*w,)
            cur_tokens = w_tokens.view(-1)[beam_idxs]  # (batch_sze*w,)
            cur_lls = w_logprobs.view(-1)[beam_idxs]  # (batch_size*w,)

            beam = beam.repeat(1, w).view(batch_size*w*w, cur_len)[beam_idxs]  # (batch_size*w, cur_len)
            beam = torch.cat([beam, cur_tokens.unsqueeze(-1)], -1)  # (batch_size*w, cur_len+1)
            beam_nlls = beam_nlls.repeat(1, w).view(batch_size*w*w, cur_len-ends[0])[beam_idxs]
            beam_nlls = torch.cat([beam_nlls, cur_lls.unsqueeze(-1)], -1)

        cur_len += 1
        if cur_tokens.eq(sep).sum() > 0:
            for b in range(batch_size):
                offset = b*w
                toks = cur_tokens[offset:offset+w].tolist()
                for idx, tok in enumerate(toks):
                    if tok == sep and (best_outputs[b] is None or beam_ll[b, idx] > best_ll[b]):
                        best_outputs[b] = beam[offset+idx]
                        best_nlls[b] = beam_nlls[offset+idx]
                        best_ll[b] = beam_ll[b, idx]
        if all(best_ll[b] is not None and best_ll[b] > beam_ll[b, 0] for b in range(batch_size)):
            break

    outputs = [{} for _ in range(batch_size)]
    for b, output in enumerate(outputs):
        output['context'] = context_cpu[b].tolist()
        output['ended'] = best_outputs[b] is not None
        output['tokens'] = (best_outputs[b] if best_outputs[b] is not None else beam[w*b]).tolist()
        output['tokens'] = output['tokens'][len(output['context']):]
        output['nll4tok'] = (best_nlls[b] if best_nlls[b] is not None else beam_nlls[w*b]).tolist()
        output['nll4tok'] = [-x for x in output['nll4tok'][1:]]
        output['ppl4tok'] = [np.exp(nll) for nll in output['nll4tok']]
        output['ppl'] = np.exp(sum(output['nll4tok'])/len(output['nll4tok']))
        output['len'] = len(output['tokens'])

    return outputs

def gumbel_like(*args, **kwargs):
    return _gumbel(torch.rand_like(*args, **kwargs))


def gumbel(*args, **kwargs):
    return _gumbel(torch.rand(*args, **kwargs))


def _gumbel(u):
    return -torch.log(-torch.log(u))


def gumbel_with_maximum(phi, T, dim=-1):
    """
    Samples a set of gumbels which are conditioned on having a maximum along a dimension
    phi.max(dim)[0] should be broadcastable with the desired maximum T
    """
    # Gumbel with location phi
    g_phi = phi + gumbel_like(phi)
    Z, argmax = g_phi.max(dim)
    g = _shift_gumbel_maximum(g_phi, T, dim, Z=Z)
    # CHECK_VALIDITY = True
    # if CHECK_VALIDITY:
    #     g_inv = _shift_gumbel_maximum(g, Z, dim)
    #     assert (((g_phi - g_inv) < 1e-3) | (g_phi == g_inv)).all()
    return g, argmax


def _shift_gumbel_maximum(g_phi, T, dim=-1, Z=None):
    if Z is None:
        Z, _ = g_phi.max(dim)
    u = T.unsqueeze(dim) - g_phi + torch.log1p(-torch.exp(g_phi - Z.unsqueeze(dim)))
    return T.unsqueeze(dim) - F.relu(u) - torch.log1p(torch.exp(-u.abs()))

def gumbel_sbs_decode(model, init, w, max_len, sep, device, batch_size):
    if init is None:
        context = torch.full((batch_size, 1), sep, dtype=torch.long, device=device)
        ends = torch.zeros_like(context[:, 0])
    else:
        context, ends = init
    assert (all([ends[i] == ends[0] for i in range(len(ends))]))
    cur_len = ends[0] + 1
    batch_size = context.size(0)
    context_cpu = context.cpu()

    beam = context.repeat(w, 1, 1).transpose(0, 1).reshape(batch_size * w, cur_len)
    beam_offset = (torch.arange(batch_size) * w).repeat(w, 1).t().reshape(-1).to(device)
    beam_nlls = torch.zeros(batch_size * w, 1, device=device)
    beam_ll = torch.zeros(batch_size, w, device=device)
    best_outputs, best_ll, best_gumbel, best_nlls = [[None for _ in range(batch_size)] for _ in range(4)]

    for i in trange(max_len):
        if i == 0:
            logits = model(context)[0][:, -1, :]  # (batch_size, V)
            logprobs = F.log_softmax(logits, -1)  # (batch_size, V)
            gumbel = gumbel_like(logprobs) + logprobs  # (batch_size, V)
            z, _ = gumbel.max(dim=-1, keepdims=True)  # (batch_size, 1)
            gumbel_tilde = -(-torch.exp(-z)+torch.exp(-gumbel)+1.0).log()  # (batch_size, V), +1.0 is exp(-G_phi_N)

            beam_gumbel, w_tokens = torch.topk(gumbel_tilde, w)  # (batch_size, w)
            cur_tokens = w_tokens.view(-1)  # (batch_size*w,)
            cur_lls = logprobs.gather(-1, w_tokens).view(-1)  # (batch_size*w,)

        else:
            logits = model(beam)[0][:, -1, :]  # (batch_size*w, V)
            V = logits.size(-1)
            logprobs = F.log_softmax(logits, -1)  # (batch_size*w, V)
            gumbel = Gumbel(loc=logprobs+beam_ll.view(batch_size*w, 1), scale=1.0).sample()  # (batch_size*w, V)
            z, _ = gumbel.max(dim=-1, keepdims=True)  # (batch_size*w, 1)
            gumbel_tilde, _ = gumbel_with_maximum(
                logprobs+beam_ll.view(batch_size*w, 1),  # (batch_size*w, V)
                beam_gumbel.view(batch_size*w), # (batch_size*w,)
                dim=-1)  # gumbel_tilde: (batch_size*w, V)

            beam_gumbel, beam_idxs = torch.topk(gumbel_tilde.view(batch_size, w*V), w)  # (batch_size, w), beam_idxs in [0,w*V)
            beam = beam[beam_idxs.view(-1)//V + beam_offset]  # (batch_size*w, cur_len)
            cur_tokens = (beam_idxs % V).view(-1)  # (batch_size*w,)
            cur_lls = logprobs.view(batch_size, w*V).gather(-1, beam_idxs).view(-1)  # (batch_size*w,)

        beam = torch.cat([beam, cur_tokens.unsqueeze(-1)], -1)  # (batch_size*w, cur_len+1)
        beam_nlls = torch.cat([beam_nlls, cur_lls.unsqueeze(-1)], -1)
        beam_ll += cur_lls.view(batch_size, w)
        cur_len += 1

        if cur_tokens.eq(sep).sum() > 0:
            for b in range(batch_size):
                offset = b * w
                toks = cur_tokens[offset:offset + w].tolist()
                for idx, tok in enumerate(toks):
                    if tok == sep and (best_outputs[b] is None or beam_gumbel[b, idx] > best_gumbel[b]):
                        best_outputs[b] = beam[offset + idx]
                        best_nlls[b] = beam_nlls[offset + idx]
                        best_ll[b] = beam_ll[b, idx]
                        best_gumbel[b] = beam_gumbel[b, idx]
        if all(best_ll[b] is not None and best_ll[b] > beam_ll[b, 0] for b in range(batch_size)):
            break

    outputs = [{} for _ in range(batch_size)]
    for b, output in enumerate(outputs):
        output['context'] = context_cpu[b].tolist()
        output['ended'] = best_outputs[b] is not None
        output['tokens'] = (best_outputs[b] if best_outputs[b] is not None else beam[w * b]).tolist()
        output['tokens'] = output['tokens'][len(output['context']):]
        output['nll4tok'] = (best_nlls[b] if best_nlls[b] is not None else beam_nlls[w * b]).tolist()
        output['nll4tok'] = [-x for x in output['nll4tok'][1:]]
        output['ppl4tok'] = [np.exp(nll) for nll in output['nll4tok']]
        output['ppl'] = np.exp(sum(output['nll4tok']) / len(output['nll4tok']))
        output['len'] = len(output['tokens'])

    return outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True, type=str,
                        help="file to write output to")
    parser.add_argument("--context_path", default=None, type=str,
                        help="file with jsonl contexts")
    parser.add_argument('--cache_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='gpt2-large',
                        help='pretrained model name')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--tokens', action='store_true')
    parser.add_argument('--greedy', action='store_true',
                        help='greedy decoding')
    parser.add_argument('-t', type=float, default=None,
                        help='Temperature for sampling')
    parser.add_argument('-k', type=int, default=None,
                        help='k for top-k sampling')
    parser.add_argument('-p', type=float, default=None,
                        help='p for Nucleus (top-p) sampling')
    parser.add_argument('-m', type=float, default=None,
                        help='mass of original dist to interpolate')
    parser.add_argument('-w', type=int, default=None,
                        help='width for beam search')
    parser.add_argument("--gumbel", action='store_true',
                        help="use gumbel stochastic beam search")
    parser.add_argument('-n', type=int, default=5000,
                        help='how many samples to produce')
    parser.add_argument('--fixed_length', action='store_true',
                        help='if doing beam search, use this for fixed-length decoding')
    parser.add_argument('--max_len', type=int, default=200,
                        help='maximum length of generation')
    parser.add_argument('--gpu', type=int, default=0,
                        help="Which GPU to run on")
    parser.add_argument('--skip', type=int, default=0,
                        help='skip first n lines')
    parser.add_argument('--fin', type=int, default=None,
                        help='maximum number of batches to complete')
    args = parser.parse_args()

    if args.seed is None:
        import time
        millis = int(round(time.time() * 1000))
        args.seed = millis

    print(args)


    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger = logging.getLogger(__name__)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu") # pylint: disable=no-member
    n_gpu = torch.cuda.device_count()

    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    assert(not (args.k and args.p))
    with open(args.output_path, 'w'):
        pass

    # Load tokenizer and model
    # This loading functions also add new tokens and embeddings called `special tokens`
    # These new embeddings will be fine-tuned on the RocStories dataset
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, do_lower_case=True)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.to(device)
    SEP = tokenizer.encode(tokenizer.bos_token)[0]

    def tokenize_and_encode(obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, int):
            return obj
        return list(tokenize_and_encode(o) for o in obj)

    # Compute the max input length for the Transformer
    max_length = args.max_len

    if args.context_path is not None:
        logger.info("Encoding dataset...")
        if args.cache_path is not None and os.path.exists(args.cache_path):
            dataset = torch.load(args.cache_path, map_location=device)
        else:
            dataset = load_dataset(args.context_path, args.batch_size, device, bs=args.w is not None)

        if args.cache_path is not None and not os.path.exists(args.cache_path):
            torch.save(dataset, args.cache_path)
    else:
        dataset = [ None for _ in range(args.n // args.batch_size) ]

    model.eval()
    outputs = []
    writer = open(args.output_path, "w")
    try:
        for b, batch in enumerate(tqdm(dataset[args.skip:args.fin], desc="Generating")):
            with torch.no_grad():
                if args.w is None:
                    output = decode(model, args.batch_size, max_length, SEP, device, 
                                temp=args.t, k=args.k, p=args.p, greedy=args.greedy,
                                m=args.m, init=batch)
                else:
                    if args.gumbel:
                        output = gumbel_sbs_decode(model, batch, args.w, max_length, SEP, device, args.batch_size)
                    else:
                        output = bs_decode_simplified(model, batch, args.w, max_length, SEP, device)
                outputs.extend(output)
                for o in output:
                    o['string'] = tokenizer.decode(o['tokens'])
                    print(json.dumps(o), file=writer, flush=True)
    except (KeyboardInterrupt, SystemExit):
        pass

    writer.close()

if __name__ == '__main__':
    main()
