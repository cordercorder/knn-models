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


logger = logging.getLogger("knn_models_cli.dedup_datastore")


def main(args: Namespace):
    keys_dtype = np.float16 if args.keys_dtype == "fp16" else np.float32

    datastore_keys_path = os.path.join(args.datastore, "keys.npy")
    logger.info(f"Loading datastore keys from {datastore_keys_path}")
    datastore_keys = np.memmap(
        datastore_keys_path,
        dtype=keys_dtype,
        mode="r",
        shape=(args.datastore_size, args.keys_dimension)
    )

    datastore_values_path = os.path.join(args.datastore, "values.npy")
    logger.info(f"Loading datastore values from {datastore_keys_path}")
    datastore_values = np.memmap(
        datastore_values_path,
        dtype=np.int64,
        mode="r",
        shape=(args.datastore_size, )
    )

    # there may identical keys with different values
    # we ignore this case for simplicity
    indices = np.unique(datastore_keys, return_index=True, axis=0)[1]
    indices = np.sort(indices)

    deduped_datastore_keys_path = os.path.join(args.deduped_datastore, "keys.npy")
    logger.info(f"Saving deduped datastore keys to {deduped_datastore_keys_path}")
    deduped_datastore_keys = np.memmap(
        deduped_datastore_keys_path,
        dtype=keys_dtype,
        mode="w+",
        shape=(indices.shape[0], args.keys_dimension)
    )
    deduped_datastore_keys[:] = datastore_keys[indices]
    deduped_datastore_keys.flush()

    deduped_datastore_values_path = os.path.join(args.deduped_datastore, "values.npy")
    logger.info(f"Saving deduped datastore values to {deduped_datastore_values_path}")
    deduped_datastore_values = np.memmap(
        deduped_datastore_values_path,
        dtype=np.int64,
        mode="w+",
        shape=(indices.shape[0], )
    )
    deduped_datastore_values[:] = datastore_values[indices]
    deduped_datastore_values.flush()

    logger.info(f"Deduped datastore size: {indices.shape[0]}")
    logger.info("Deduplicate datastore complete")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datastore", type=str, required=True, help="path to datastore directory")
    parser.add_argument("--datastore-size", type=int, required=True, help="the number of tokens in datastore")
    parser.add_argument("--keys-dtype", type=str, choices=["fp16", "fp32"], default="fp16", help="keys dtype of the datastore")
    parser.add_argument("--keys-dimension", required=True, type=int, help="the feature dimension of datastore keys")

    parser.add_argument("--deduped-datastore", type=str, required=True, help="path to the deduped datastore directory")
    return parser


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
