import sys
import argparse


def _get_ngram_freq(input, output, n):
    assert n > 0, "n must be larger than 0"

    counter = {}
    
    for tokens in input:
        tokens = tokens.strip()
        tokens = tokens.split()
        num_tokens = len(tokens)
        for i in range(0, num_tokens - n):
            ngram = tuple(tokens[i: i + n])
            counter[ngram] = counter.get(ngram, 0) + 1
    
    for ngram, freq in sorted(counter.items(), key=lambda item: item[1], reverse=True):
        ngram = " ".join(ngram)
        print(f"{ngram}\t{freq}", file=output)
    
    output.close()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=argparse.FileType(mode="r", encoding="utf-8"), default=sys.stdin)
    parser.add_argument("--output", type=argparse.FileType(mode="w", encoding="utf-8"), default=sys.stdout)

    parser.add_argument("--n", type=int, default=1)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    _get_ngram_freq(args.input, args.output, args.n)


if __name__ == "__main__":
    main()
