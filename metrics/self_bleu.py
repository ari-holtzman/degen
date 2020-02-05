"""
Usage:
$ python metrics/self_bleu.py gen.jsonl
"""
import argparse
import json
import os
import random
from functools import partial
from multiprocessing.pool import Pool

import spacy
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--n_sample", type=int, default=1000,
                        help="how many sentences to sample to calculate bleu")
    parser.add_argument("--logto", type=str)

    return parser.parse_args()


def bleu_i(weights, all_sentences, smoothing_function, i):
    # noinspection PyTypeChecker
    return sentence_bleu(
        references=all_sentences[:i] + all_sentences[i + 1:],
        hypothesis=all_sentences[i],
        weights=weights,
        smoothing_function=smoothing_function)


def main():
    args = parse_args()
    random.seed(0)
    nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    all_sentences = []
    with open(args.file, "r") as f_in:
        for line in f_in:
            obj = json.loads(line.strip())
            gen = obj['tokens']
            all_sentences.append(gen)
    smoothing_function = SmoothingFunction().method1

    pool = Pool(processes=os.cpu_count())
    bleu_scores = []
    for n_gram in range(1, 6):

        if n_gram == 1:
            weights = (1.0, 0, 0, 0)
        elif n_gram == 2:
            weights = (0.5, 0.5, 0, 0)
        elif n_gram == 3:
            weights = (1.0 / 3, 1.0 / 3, 1.0 / 3, 0)
        elif n_gram == 4:
            weights = (0.25, 0.25, 0.25, 0.25)
        elif n_gram == 5:
            weights = (0.2, 0.2, 0.2, 0.2, 0.2)
        else:
            raise ValueError
        bleu_scores.append(
            list(tqdm(
                pool.imap_unordered(
                    partial(bleu_i, weights, all_sentences, smoothing_function),
                    random.sample(range(len(all_sentences)), args.n_sample)),
                total=args.n_sample,
                smoothing=0.0,
                desc=f"bleu-{n_gram}")))
        print(f"\n\nbleu-{n_gram} = {sum(bleu_scores[n_gram - 1]) / args.n_sample}")

    for n_gram in range(5):
        print(f"bleu-{n_gram + 1} = {sum(bleu_scores[n_gram]) / args.n_sample}")

    if args.logto:
        with open(args.logto, 'a') as fout:
            print(f"{os.path.basename(args.file)}", end='\t', file=fout)
            for n_gram in range(5):
                print(f"{sum(bleu_scores[n_gram]) / args.n_sample}", end='\t', file=fout)
            print(file=fout)


if __name__ == '__main__':
    main()

