import argparse, csv, emoji

# Mechanical Turk can't handle emojis
def remove_emojis(s):
  return ''.join(filter(lambda c: c not in emoji.UNICODE_EMOJI, s))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str,
                        help="one-article per line file")
    parser.add_argument("output_path", type=str,
                        help="file to write output CSV to")
    parser.add_argument('-b', type=int, default=25,
                        help='chunk size')
    args = parser.parse_args()
    print(args)

    with open(args.input_path, 'r') as input_file, open(args.output_path, 'w') as output_file:
        writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
        writer.writerow([ 'OUT%d' % i for i in range(args.b)])
        cols = []
        i = 0
        for line in input_file:
            if i > 0 and (i % args.b) == 0:
                writer.writerow(cols)
                cols = []
            cols.append(remove_emojis(line.strip().replace('<|endoftext|>', '')))
            i += 1
        if i > 0 and (i % args.b) == 0:
            writer.writerow(cols)
            cols = []

if __name__ == '__main__':
    main()
