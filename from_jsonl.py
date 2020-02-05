"""
  Batch extract attributes from jsonl files.
"""
import argparse, json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str,
                        help="jsonl to extract from")
    parser.add_argument('key', type=,str,
                        help='key to extract from each json')
    parser.add_argument("--output_path", type=str, default=None,
                        help="file to write output to; when not set the name of input_path is combined with the key being used")
    args = parser.parse_args()
    print(args)

    output_path = args.output_path if args.output_path else '%s_%s.txt' % (args.input_path[:-6], args.key)
    with open(args.input_path, 'r') as input_file, open(output_path, 'w') as output_file:
        for json_str in input_file:
            j = json.loads(json_str.strip())
            if args.key == 'string':
                output_file.write(j['string'].strip().replace('\n', '<br>') + '\n')
            elif args.key == 'tokens':
                output_file.write(' '.join(map(str, j['tokens'])) + '\n')
            elif args.key == 'ppl':
                output_file.write('%f\n' % j['ppl'])
            else:
                raise NotImplementedError('Extracting this property has not been implemented!')

if __name__ == '__main__':
    main()
