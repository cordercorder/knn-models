import random
import argparse

from itertools import zip_longest


def _sample_parallel_corpus(
    src_path,
    tgt_path,
    output_src_path,
    output_tgt_path,
    sampled_sent_num,
    keep_order,
    seed,
):
    random.seed(seed)

    src_fin = open(src_path, mode="r", encoding="utf-8")
    tgt_fin = open(tgt_path, mode="r", encoding="utf-8")

    src_data = []
    tgt_data = []

    for src_line, tgt_line in zip_longest(src_fin, tgt_fin, fillvalue=None):
        assert src_line is not None and tgt_line is not None
        src_line = src_line.strip()
        tgt_line = tgt_line.strip()
        src_data.append(src_line)
        tgt_data.append(tgt_line)
    
    num_sents = len(src_data)

    sampled_indices = random.sample(range(num_sents), k=sampled_sent_num)

    if keep_order:
        sampled_indices.sort()

    sampled_src_data = []
    sampled_tgt_data = []

    for idx in sampled_indices:
        sampled_src_data.append(src_data[idx])
        sampled_tgt_data.append(tgt_data[idx])

    with open(output_src_path, mode="w", encoding="utf-8") as src_fout, \
        open(output_tgt_path, mode="w", encoding="utf-8") as tgt_fout:

        for src_line, tgt_line in zip(sampled_src_data, sampled_tgt_data):
            src_fout.write(src_line + "\n")
            tgt_fout.write(tgt_line + "\n")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--src-path", type=str, required=True)
    parser.add_argument("--tgt-path", type=str, required=True)

    parser.add_argument("--output-src-path", type=str, required=True)
    parser.add_argument("--output-tgt-path", type=str, required=True)

    parser.add_argument("--sampled-sent-num", type=int, required=True)
    parser.add_argument("--keep-order", action="store_true", default=False)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    _sample_parallel_corpus(
        args.src_path,
        args.tgt_path,
        args.output_src_path,
        args.output_tgt_path,
        args.sampled_sent_num,
        args.keep_order,
        args.seed,
    )


if __name__ == "__main__":
    main()
