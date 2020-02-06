# Degen
The Official Repository for "The Curious Case of Neural Text Degeneration"

If you want to use Nucleus Sampling, you can use the implementation in [Hugging Face Transformers](https://github.com/huggingface/transformers/blob/master/examples/run_generation.py) with many pretrained models, including GPT-2!

# Generations

All conditional and unconditional generations are available [here](https://drive.google.com/file/d/1add49ypQLPC8ddGAbLcQVVfXnwAXkkMz/view?usp=sharing). 

# Requirements

`pytorch` must be installed.

The other required modules are in `requirements.txt` and can be installed with:

```
pip install -r requirements.txt
```

# Generating Your Own

Use `gen.py` to generate:
```
python gen.py --model_name gpt2-large --batch_size 10 -n 50 --output_path output.jsonl --gpu 0 -p 0.95 --seed 0
```
`--help` will show options for decoding strategies and parameters


# Formatting Data

For conditional generations, you'll need to format contexts.

Use `encode_jsonl.py` to tokenize data from https://github.com/openai/gpt-2-output-dataset so it can be be used for conditional generation:
```
python encode_jsonl.py raw.jsonl tokenized.jsonl
```
`--help` will show more options


Use `filter_for_conditional.py` for creating contexts for conditional generations:
```
python filter_for_conditional.py tokenized.jsonl filtered.jsonl
```
`--help` will show more options


Now use `sort_jsonl_by_length.py` to sort things for more efficient batching:
```
python sort_jsonl_by_length.py filtered.jsonl sorted.jsonl
```

Finally, if you'd like to use Beam Search or Stochastic Beam Search. First create a cache file by generating for another algorithm for a non-beam decoding algorithm with the `--cache` flag:
```
python gen.py --model_name gpt2-large --batch_size 10 -n 50 --context_path sorted.jsonl --output_path output.jsonl --gpu 0 -k 40 --seed 0 --cache first.cache
```

Next, we'll reprocess the cache file for Beam Search:
```
python rebatch_inits_for_beamsearch.py first.cache --batch_size 4 --out bs_4.cache
```

Now we can decode with Beam Search:
```
python gen.py --model_name gpt2-large --batch_size 4 -n 40 --context_path sorted.jsonl --cache bs_4.cache --output_path output.jsonl --gpu 0 -w 4
```

# MTurk

`mturk_form.html` is the Amazon Mechanical Turk template we used for experiments.

Use `from_jsonl.py` to extract strings (and other attributes from jsonl files):
```
python from_jsonl.py output.jsonl string output_string.txt
```
`--help` will show more options

`chunk4turk.py` can be used to batch texts to be used with the above MTurk template:
```
python chunk4turk.py output_string.txt output_mturk.csv
```
`--help` will show more options
