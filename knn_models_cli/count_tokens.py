import argparse

from fairseq.data import data_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-bin-path", "-d", required=True, type=str)
    args = parser.parse_args()
    
    dataset = data_utils.load_indexed_dataset(args.data_bin_path, None)
    print(dataset.sizes.sum())


if __name__ == "__main__":
    main()
