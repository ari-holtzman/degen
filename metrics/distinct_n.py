import argparse
import json
import os
import logging
from collections import Counter

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("N", type=int, help="N in distinct-N metric")
    parser.add_argument("--numbers-only", action="store_true")
    return parser.parse_args()

def distinct_n(examples, n):
    counter = Counter()
    n_total = 0
    n_distinct = 0
    for example in examples:
        gen = example['tokens']
        for token in zip(*(gen[i:] for i in range(n))):
            if token not in counter:
                n_distinct += 1
            elif counter[token] == 1:
                n_distinct -= 1
            counter[token] += 1
            n_total += 1
    return n_distinct, n_total

def main():
    args = parse_args()
    with open(args.file, "r") as fin:
        examples = [json.loads(l.strip()) for l in fin]

    n_distinct, n_total = distinct_n(examples, args.N)
    if not args.numbers_only:
        print(f"filename\tdistinct {args.N}-grams\ttotal {args.N}-grams\tdistinct proportion")
    print(f"{os.path.basename(args.file)}\t{n_distinct}\t{n_total}\t{n_distinct/n_total}")


if __name__ == '__main__':
    main()
