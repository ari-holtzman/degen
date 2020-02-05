"""
Usage:
$ python metrics/zipf.py gen.jsonl
"""
import argparse
import json
import operator
import os
from collections import Counter
import numpy as np
from scipy import stats

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("file", type=str)
    parser.add_argument("--N", type=int, default=5000)
    parser.add_argument("--numbers-only", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    cnt = Counter()
    if not args.numbers_only:
        print("filename\tzipf value\tregression r value \tregression p value")
    print(f"{os.path.basename(args.file)}", end="\t")

    with open(args.file, 'r') as fin:
        for l in fin:
            obj = json.loads(l.strip())
            gen = obj['tokens']
            cnt.update(gen)

    xs = np.arange(1, min(len(cnt), args.N)+1)
    ys = np.array(sorted(cnt.values(), key=operator.neg)[:args.N])
    a, b, r, p, std = stats.linregress(np.log(xs), np.log(ys))
    print(f"{-a}\t{-r}\t{p}")


if __name__ == '__main__':
    main()
