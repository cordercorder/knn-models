import os
import sys
import logging
import argparse

from knn_models.dim_reduce_utils import (
    pca_dimension_reduction,
)


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)


logger = logging.getLogger("knn_models_cli.reduce_datastore_dims")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["PCA"], required=True, 
                        help="method used for dimension reduction")
    
    parser.add_argument("--datastore", required=True, type=str, 
                        help="path to datastore directory")
    parser.add_argument("--datastore-size", required=True, type=int, 
                        help="the number of tokens in datastore")
    parser.add_argument("--keys-dimension", required=True, type=int, 
                        help="the feature dimension of datastore keys")
    parser.add_argument("--keys-dtype", default="fp16", choices=["fp16", "fp32"], type=str, 
                        help="keys dtype of the datastore")
    
    parser.add_argument("--transformed-datastore", required=True, type=str, 
                        help="path to the datastore after dimension reduction")
    
    args, unknown = parser.parse_known_args()

    if args.method == "PCA":
        parser.add_argument("--reduced-keys-dimension", required=True, type=int, 
                            help="reduced feature dimension of datastore keys")
        parser.add_argument("--random-rotation", action="store_true", default=False, 
                            help="whether to perform random rotation before PCA")
    return parser


def cli_main():
    parser = get_parser()
    args = parser.parse_args()

    if args.method == "PCA":
        # validate args
        assert args.reduced_keys_dimension < args.keys_dimension, \
            "reduced feature dimension should be small than original feature dimension"
        pca_dimension_reduction(
            args.datastore,
            args.datastore_size,
            args.keys_dimension,
            args.keys_dtype,
            args.reduced_keys_dimension,
            args.random_rotation,
            args.transformed_datastore,
        )


if __name__ == "__main__":
    cli_main()
