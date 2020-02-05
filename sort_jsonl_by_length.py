import argparse, json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str,
                        help="jsonl to extract from")
    parser.add_argument("output_path", type=str,
                        help="jsonl to extract from")
    args = parser.parse_args()
    print(args)

    js = []
    with open(args.input_path, 'r') as input_file:
        for json_str in input_file:
            j = json.loads(json_str.strip())
            js.append(j)
    js = sorted(js, key=lambda j: len(j['tokens']))
    with open(args.output_path, 'w') as out:
        for j in js:
            out.write(json.dumps(j) + '\n')

if __name__ == '__main__':
    main()
