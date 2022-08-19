import os
import sys
import math
import logging
import argparse

from knn_models.dim_reduce_utils import (
    train_pckmt,
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
    parser.add_argument("--method", type=str, choices=["PCA", "PCKMT"], required=True, 
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
    
    parser.add_argument("--reduced-keys-dimension", required=True, type=int, 
                        help="reduced feature dimension of datastore keys")
    
    args, unknown = parser.parse_known_args()

    if args.method == "PCA":
        parser.add_argument("--random-rotation", action="store_true", default=False, 
                            help="whether to perform random rotation before PCA")
    elif args.method == "PCKMT":
        parser.add_argument("--compact-net-hidden-size", type=int, default=None, 
                            help="hidden size of compact network. "
                            "it will be set to `keys_dimension`/4 by default")

        parser.add_argument("--stage", type=str, choices=["train_pckmt", "apply_pckmt"], required=True, 
                            help="`train_pckmt`: train compact network. "
                            "`apply_pckmt`: use the trained compact network for dimension reduction")
        
        args, unknown = parser.parse_known_args()

        if args.stage == "train_pckmt":
            parser.add_argument("--seed", type=int, default=1, 
                                help="random seed")

            parser.add_argument("--compact-net-dropout", type=float, default=0.0)

            parser.add_argument("--num-trained-keys", default=math.inf, type=int, 
                                help="maximum number of keys used for training compact network")
            parser.add_argument("--vocab-size", type=int ,required=True, 
                                help="number of unique tokens in the vocabulary")
            
            parser.add_argument("--log-interval", type=int, default=100, 
                                help="print the training progress every `log_interval` updates")
            
            parser.add_argument("--batch-size", type=int, default=1024, 
                                help="number of keys in one batch")
            parser.add_argument("--num-workers", type=int, default=1, 
                                help="number of subprocess used for data loading")

            parser.add_argument("--max-update", type=int, default=0, 
                                help="maximum training steps")
            parser.add_argument("--max-epoch", type=int, default=0, 
                                help="maximum training epochs")
            parser.add_argument("--update-freq", type=int, default=1, 
                                help="update parameters every `update_freq` batches")
            
            parser.add_argument("--lr", type=float, default=0.0005,
                                help="learning rate")
            parser.add_argument("--betas", type=eval, default="(0.9, 0.999)", 
                                help="betas for Adam optimizer")
            parser.add_argument("--weight-decay", type=float, default=0.0, 
                                help="weight decay")
            parser.add_argument("--clip-norm", type=float, default=0.0, 
                                help="clip threshold of gradients")

            parser.add_argument("--dbscan-eps", type=float, default=10, 
                                help="the maximum distance between two samples for one to be "
                                "considered as in the neighborhood of the other in DBSCAN")
            parser.add_argument("--dbscan-min-samples", type=int, default=4, 
                                help="the number of samples (or total weight) in a neighborhood "
                                "for a point to be considered as a core point in DBSCAN")
            parser.add_argument("--dbscan-max-samples", type=int, default=100000, 
                                help="maximum number of samples to train DBSCAN")
    return parser


def cli_main():
    parser = get_parser()
    args = parser.parse_args()

    # validate args
    assert args.reduced_keys_dimension < args.keys_dimension, \
        "reduced feature dimension should be smaller than original feature dimension"

    if args.method == "PCA":
        pca_dimension_reduction(
            args.datastore,
            args.datastore_size,
            args.keys_dimension,
            args.keys_dtype,
            args.reduced_keys_dimension,
            args.random_rotation,
            args.transformed_datastore,
        )
    elif args.method == "PCKMT":
        if args.stage == "train_pckmt":
            train_pckmt(
                args.datastore,
                args.datastore_size,
                args.keys_dimension,
                args.keys_dtype,
                args.reduced_keys_dimension,
                args.compact_net_hidden_size,
                args.compact_net_dropout,
                args.num_trained_keys,
                args.batch_size,
                args.num_workers,
                args.vocab_size,
                args.log_interval,
                args.max_update,
                args.max_epoch,
                args.update_freq,
                args.lr,
                args.betas,
                args.weight_decay,
                args.clip_norm,
                args.dbscan_eps,
                args.dbscan_min_samples,
                args.dbscan_max_samples,
                args.seed,
                args.transformed_datastore,
            )


if __name__ == "__main__":
    cli_main()
