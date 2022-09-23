import os
import sys
import logging
import argparse
import numpy as np

from argparse import Namespace


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)

logger = logging.getLogger("knn_models_cli.merge_datastore")


def main(args: Namespace):
    merged_datastore_size = sum(args.datastore_size_list)

    keys_dtype_table = {"fp16": np.float16, "fp32": np.float32}

    merged_datastore_keys_path = os.path.join(args.merged_datastore, "keys.npy")
    logger.info(f"Saving merged datastore keys to {merged_datastore_keys_path}")
    merged_keys_dtype = keys_dtype_table[args.merged_keys_dtype]
    merged_datastore_keys = np.memmap(
        merged_datastore_keys_path,
        dtype=merged_keys_dtype,
        mode="w+",
        shape=(merged_datastore_size, args.keys_dimension)
    )

    merged_datastore_values_path = os.path.join(args.merged_datastore, "values.npy")
    logger.info(f"Saving merged datastore values to {merged_datastore_values_path}")
    merged_datastore_values = np.memmap(
        merged_datastore_values_path,
        dtype=np.int64,
        mode="w+",
        shape=(merged_datastore_size, )
    )

    prev_idx = 0
    for datastore, datastore_size, keys_dtype in zip(args.datastore_list, args.datastore_size_list, args.keys_dtype_list):
        current_idx = prev_idx + datastore_size

        datastore_keys_path = os.path.join(datastore, "keys.npy")
        logger.info(f"Loading datastore keys from {datastore_keys_path}")
        datastore_keys = np.memmap(
            datastore_keys_path,
            dtype=keys_dtype_table[keys_dtype],
            mode="r",
            shape=(datastore_size, args.keys_dimension)
        )
        merged_datastore_keys[prev_idx: current_idx] = datastore_keys.astype(merged_keys_dtype)

        datastore_values_path = os.path.join(datastore, "values.npy")
        logger.info(f"Loading datastore values from {datastore_values_path}")
        datastore_values = np.memmap(
            datastore_values_path,
            dtype=np.int64,
            mode="r",
            shape=(datastore_size, )
        )

        merged_datastore_values[prev_idx: current_idx] = datastore_values

        prev_idx = current_idx
    
    merged_datastore_values.flush()

    logger.info(f"Merged datastore size: {merged_datastore_size}")
    logger.info("Merge datastore complete")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datastore-list", type=str, nargs="+", required=True, help="space separated datastore path")
    parser.add_argument("--datastore-size-list", type=int, nargs="+", required=True, help="space separeted datastore size")
    parser.add_argument("--keys-dtype-list", type=str, nargs="+", choices=["fp16", "fp32"], required=True, help="space separated datastore keys dtype")
    parser.add_argument("--keys-dimension", required=True, type=int, help="the feature dimension of datastore keys")

    parser.add_argument("--merged-datastore", type=str, required=True, help="path to the merged datastore directory")
    parser.add_argument("--merged-keys-dtype", type=str, choices=["fp16", "fp32"], default="fp16", help="keys dtype of the merged datastore")
    return parser


def validate_args(args: Namespace):
    assert len(args.datastore_list) == len(args.datastore_size_list) == len(args.keys_dtype_list), \
        f"The length of `datastore_list`, `datastore_size_list` and `keys_dtype_list` must be the same." \
        f"Length of `datastore_list`: {args.datastore_list}, length of `datastore_size_list`: {args.datastore_size_list}, " \
        f"length of `keys_dtype_list`: {args.keys_dtype_list}."


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    validate_args(args)
    main(args)


if __name__ == "__main__":
    cli_main()
