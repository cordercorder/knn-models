import os
import sys
import logging
import argparse

from knn_models.prune_utils import (
    random_pruning,
    greedy_merge_pruning,
)


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)


logger = logging.getLogger("knn_models_cli.prune_datastore")


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1, 
                        help="random seed")

    parser.add_argument("--method", type=str, choices=["random_pruning", "greedy_merge"], required=True)

    parser.add_argument("--datastore", required=True, type=str, 
                        help="path to datastore directory")
    parser.add_argument("--datastore-size", required=True, type=int, 
                        help="the number of tokens in datastore")
    parser.add_argument("--keys-dimension", required=True, type=int, 
                        help="the feature dimension of datastore keys")
    parser.add_argument("--keys-dtype", default="fp16", choices=["fp16", "fp32"], type=str, 
                        help="keys dtype of the datastore")

    parser.add_argument("--pruned-datastore", required=True, type=str, 
                        help="path to the pruned datastore directory")

    args, unknown = parser.parse_known_args()

    if args.method == "greedy_merge":
        parser.add_argument("--use-gpu", action="store_true", default=False, 
                            help="whether to use GPU to train the faiss index")

        parser.add_argument("--batch-size", default=1024, type=int, 
                            help="number of keys in one batch")

        parser.add_argument("--num-neighbors", type=int, default=8, 
                            help="the number of neighbors to retrieve")
        parser.add_argument("--nprobe", type=int, default=32, 
                            help="number of clusters to query")
        parser.add_argument("--knn-fp16", action="store_true", default=False, 
                            help="whether to perform intermediate calculations in float16")

        parser.add_argument("--save-knn-distance", action="store_true", default=False, 
                            help="whether to save the distance between the key and its nearest neighbors")
                            
        parser.add_argument("--log-interval", type=int, default=100, 
                            help="print the progress in an interval "
                            "during retrieving k-nearest neighbors for each keys in the datastore")
    elif args.method == "random_pruning":
        parser.add_argument("--pruned-datastore-size", type=int, required=True, 
                            help="the datastore size after ramdom pruning")

    return parser


def cli_main():
    parser = get_parser()
    args = parser.parse_args()

    if args.method == "greedy_merge":
        # validate args
        if args.knn_fp16:
            assert args.use_gpu, "knn_fp16 can only be performed on GPU"

        greedy_merge_pruning(
            args.datastore,
            args.datastore_size,
            args.keys_dimension,
            args.keys_dtype,
            args.pruned_datastore,
            args.use_gpu,
            args.batch_size,
            args.num_neighbors,
            args.nprobe,
            args.knn_fp16,
            args.save_knn_distance,
            args.log_interval,
            args.seed,
        )

    elif args.method == "random_pruning":
        assert args.pruned_datastore_size <= args.datastore_size, \
            "datastore size after pruning can not be greater than current datastore size"

        random_pruning(
            args.datastore,
            args.datastore_size,
            args.keys_dimension,
            args.keys_dtype,
            args.pruned_datastore,
            args.pruned_datastore_size,
            args.seed,
        )


if __name__ == "__main__":
    cli_main()
