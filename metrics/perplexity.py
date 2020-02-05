"""
  Compute total perplexity of a jsonl file of generations.
"""
import argparse, json
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str,
                        help="jsonl that contains generations")
    args = parser.parse_args()
    print(args)

    nlls = []
    with open(args.input_path, 'r') as input_file:
        for json_str in input_file:
            if len(json_str.strip()) == 0:
                continue
            j = json.loads(json_str.strip())
            nll4tok = j['nll4tok']
            nlls += nll4tok
    n = len(nlls)
    ppl = np.exp(sum(map(lambda nll: nll / n, nlls)))
    print(ppl)

if __name__ == '__main__':
    main()
