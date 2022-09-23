import os
import sys
import logging
import argparse
import numpy as np


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)


logger = logging.getLogger("split_datastore_keys")


def _split_datastore_keys(
    seed, 
    datastore, 
    datastore_size, 
    num_partition
):
    rng = np.random.default_rng(seed)
    # shape: [datastore_size, ]
    indices = rng.permutation(datastore_size)
    partition_size = datastore_size // num_partition

    if datastore_size % num_partition != 0:
        remain = True
    else:
        remain = False

    prev_idx = 0
    for partition_idx in range(num_partition):
        partition_path = os.path.join(datastore, f"indices.partition_{partition_idx:02d}.npy")
        if partition_idx == num_partition - 1 and remain:
            partition = indices[prev_idx: ]
        else:
            cur_idx = prev_idx + partition_size
            partition = indices[prev_idx: cur_idx]

        with open(partition_path, mode="wb") as f:
            logger.info(f"Save partition {partition_idx} to {partition_path}")
            np.save(f, partition)
        
        prev_idx = cur_idx


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--datastore", type=str, required=True)
    parser.add_argument("--datastore-size", type=int ,required=True)
    parser.add_argument("--num-partition", type=int, required=True)
    return parser


def validate_args(args):
    assert args.num_partition > 1


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    validate_args(args)
    _split_datastore_keys(
        args.seed, 
        args.datastore, 
        args.datastore_size, 
        args.num_partition
    )


if __name__ == "__main__":
    cli_main()
