""" 
  Tokenize text stored in Open-AI style jsonl files (https://github.com/openai/gpt-2-output-dataset)
  Takes the attribute "text" and adds the attribute "tokens"
"""
import argparse
import os
import json
import random

from transformers import GPT2Tokenizer, cached_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str,
                        help="file with one-json-per-line contexts")
    parser.add_argument("output_path", type=str,
                        help="path to write output to")
    parser.add_argument('--model_name', type=str, default='gpt2-large',
                        help='pretrained model name, only used for tokenizer')
    parser.add_argument('--cut', type=int, default=None,
                        help='when set, acts as a maximum length')
    parser.add_argument('--cap', action='store_true',
                        help='if a generation isn\'t cut, should <|endoftext|> be appeneded to the end?')
    args = parser.parse_args()
    print(args)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, do_lower_case=True)
    SEP = tokenizer.encode(tokenizer.bos_token)[0]

    with open(args.input_path, 'r') as input_file, open(args.output_path, 'w') as output_file:
        for json_str in input_file:
            j = json.loads(json_str.strip())
            j['tokens'] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(j['text']))
            was_cut = False
            if args.cut and len(j['tokens']) > args.cut:
                j['tokens'] = j['tokens'][:args.cut]
                was_cut = True
            if args.cap and not was_cut:
                j['tokens'].append(SEP)
            output_file.write(json.dumps(j) + '\n')

if __name__ == '__main__':
    main()
