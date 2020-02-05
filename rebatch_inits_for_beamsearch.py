import argparse
import logging
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)

# usage: python scripts/rebatch_inits_for_beamsearch.py data/heldout_40.cache --batch_size 4 --out bs_4.cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("cache", type=str)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    cache = torch.load(args.cache)

    all_inits = []
    all_ends = []
    for inits, ends in cache:
        assert inits.shape[0] == len(ends)
        all_inits.extend(inits.tolist())
        all_ends.extend(ends)
    assert len(all_inits) == len(all_ends)

    if args.batch_size > 1:
        all_batches = []
        batch = []
        for init, end in zip(all_inits, all_ends):
            if len(batch) >= args.batch_size:
                all_batches.append(batch)
                batch = []
            if not batch or end == batch[-1][1]:
                batch.append((init[:(end+1)], end))
            else:
                all_batches.append(batch)
                batch = [(init[:(end+1)], end)]
        if batch:
            all_batches.append(batch)

        tensor_batches = []
        for batch in all_batches:
            inits = torch.tensor([x[0] for x in batch])
            ends = [x[1] for x in batch]
            assert inits.shape[0] <= args.batch_size
            assert inits.shape[0] == len(ends)
            tensor_batches.append((inits, ends))

        assert sum(x[0].shape[0] for x in tensor_batches) == sum(x[0].shape[0] for x in cache)
        for init, end in tensor_batches:
            assert all(x == end[0] for x in end)
    elif args.batch_size == 1:
        tensor_batches = []
        for i, (init, end) in enumerate(zip(all_inits, all_ends)):
            tensor_batches.append((
                torch.tensor(init[:end+1]).unsqueeze(0), [end]
            ))

    torch.save(tensor_batches, args.out)


if __name__ == '__main__':
    main()
