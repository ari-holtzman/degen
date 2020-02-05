"""
Usage:
$ python metrics/repetition.py gen.jsonl
When using
$ python metrics/repetition.py gen.jsonl --output
it will generate a repetition_gen.jsonl which contains detailed fields on how each
example is repeating itself, specifically the phrase the generation is repeating
and how many times it is repeated.
"""
import argparse
import json
import os

from transformers import GPT2Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("--numbers-only", action="store_true")
    parser.add_argument("--output", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large", do_lower_case=True)
    SEP = tokenizer.encode(tokenizer.bos_token)[0]

    objs = []
    max_n = 90

    with open(args.file, 'r') as fin:
        for l in fin:
            objs.append(json.loads(l.strip()))

    n_repeated_examples = 0
    for obj in objs:
        gen = obj['tokens']
        if gen[-1] == SEP:
            gen.pop(-1)
        rev_gen = list(reversed(gen))
        last_n_repeats = [0] * max_n

        for n in range(1, max_n + 1):
            n_repeat = 1
            while len(rev_gen[n*n_repeat:n*(n_repeat+1)]) == n and \
                    rev_gen[n*n_repeat:n*(n_repeat+1)] == rev_gen[:n]:
                n_repeat += 1
            last_n_repeats[n - 1] = n_repeat
        max_repeated_n = max(range(max_n), key=lambda x: last_n_repeats[x])

        if last_n_repeats[max_repeated_n] > 1 and (max_repeated_n+1 >= 3 or last_n_repeats[max_repeated_n] > 50):
            obj['repetition'] = {
                'repeated_phrase': list(reversed(rev_gen[:max_repeated_n + 1])),
                'repeated_times': last_n_repeats[max_repeated_n],
                'repeated_phrase_length': max_repeated_n + 1,
            }
            n_repeated_examples += 1
        else:
            obj['repetition'] = None

    if not args.numbers_only:
        print("filename\tnumber of repeating examples")
    print(f"{os.path.basename(args.file)}\t{n_repeated_examples}")
    if args.output:
        output_filename = os.path.join(os.path.dirname(args.file), "repetition_" + os.path.basename(args.file))
        with open(output_filename, 'w+') as fout:
            for obj in objs:
                print(json.dumps(obj), file=fout)


if __name__ == '__main__':
    main()
