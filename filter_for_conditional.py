# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
In order to avoid starting a generation in the middle of random text,
we start at the end of a paragraph, as long as the total length of text
to be conditioned on is less than 40 tokens. 40 is an arbitrary number
chosen because it is small enough that we could run a lot of experiments
with limited GPUs :)
"""
import argparse
import json

from transformers import GPT2Tokenizer

# For some reason HuggingFace only represent newlines inbetween non-whitespace tokens.
# So we hardcode this in to avoid strange, uninterpretable workarounds
NEWLINE = 198

def sublist_end_index(list1, list2):
    s1, s2 = ' '.join(map(str, list1)), ' '.join(map(str, list2))
    if s1 in s2:
        return s2[:s2.index(s1)].count(' ') + s1.count(' ') + 1
    else:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str,
                        help="file with one-jsonl-per-line contexts")
    parser.add_argument("output_path", type=str,
                        help="dir to write output to")
    parser.add_argument('--model_name', type=str, default='gpt2-large',
                        help='pretrained model name')
    parser.add_argument('-n', type=int, default=5000)
    parser.add_argument('-m', type=int, default=40)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(args)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, do_lower_case=True)


    with open(args.input_path, 'r') as input_file, open(args.output_path, 'w') as output_file:
        num = 0
        for json_str in input_file:
            j = json.loads(json_str.strip())
            j['tokens'] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(j['text']))
            idx = sublist_end_index([NEWLINE, NEWLINE], j['tokens'])
            if idx is not None and idx < args.m:
                j['tokens'] = j['tokens'][:idx]
                output_file.write(json.dumps(j) + '\n')
                num += 1
                if num >= args.n:
                    break

if __name__ == '__main__':
    main()
