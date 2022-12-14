import os
import sys
import math
import faiss
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


logger = logging.getLogger("knn_models_cli.build_faiss_index")


def get_gpu_resource(knn_fp16):
    resource = faiss.StandardGpuResources()

    cloner_options = None
    if knn_fp16:
        cloner_options = faiss.GpuClonerOptions()
        cloner_options.useFloat16 = True
    
    return resource, cloner_options


def build_ivfpq_index(
    datastore_keys,
    datastore,
    ncentroids,
    code_size,
    nbit_per_idx,
    nprobe,
    use_gpu,
    knn_fp16,
    num_trained_keys,
    batch_size,
    seed,
):
    trained_index_path = os.path.join(datastore, "faiss.trained_index")
    index = None

    if not os.path.isfile(trained_index_path):
        quantizer = faiss.IndexFlatL2(datastore_keys.shape[1])
        index = faiss.IndexIVFPQ(quantizer, datastore_keys.shape[1], ncentroids, code_size, nbit_per_idx)
        index.nprobe = nprobe

        if use_gpu:
            logger.info("Move index to gpu before training")
            resource, cloner_options = get_gpu_resource(knn_fp16)
            index = faiss.index_cpu_to_gpu(provider=resource, device=0, index=index, options=cloner_options)

        if datastore_keys.shape[0] > num_trained_keys:
            logger.info(f"Start sampling {num_trained_keys} datastore keys")
            # If the number of keys is larger than `args.num_trained_keys`
            # We sample a subset of keys to train the faiss index
            rng = np.random.default_rng(seed)
            sampeled_indices = rng.choice(np.arange(datastore_keys.shape[0]), size=num_trained_keys, replace=False)

            # As reading keys located at consecutive indices is faster, we sort the sampled indices before training
            # When the number of sampled indices is large, (eg., more than 40,000,000 keys), there is a sigificant speed up
            sampeled_indices = np.sort(sampeled_indices)

            trained_keys = datastore_keys[sampeled_indices].astype(np.float32)
        else:
            trained_keys = datastore_keys.astype(np.float32)

        logger.info("Start training faiss index")
        index.train(trained_keys)
        del trained_keys
        
        logger.info(f"Writing the trained faiss index to {trained_index_path}")
        if use_gpu:
            faiss.write_index(faiss.index_gpu_to_cpu(index), trained_index_path)
        else:
            faiss.write_index(index, trained_index_path)

    if index is None:
        index = faiss.read_index(trained_index_path)

        if use_gpu:
            resource, cloner_options = get_gpu_resource(knn_fp16)
            index = faiss.index_cpu_to_gpu(provider=resource, device=0, index=index, options=cloner_options)
    
    logger.info("Staring adding keys to the trained faiss index")
    current_idx = 0
    while current_idx < datastore_keys.shape[0]:
        start_idx = current_idx
        end_idx = min(start_idx + batch_size, datastore_keys.shape[0])
        index.add_with_ids(datastore_keys[start_idx: end_idx].astype(np.float32), np.arange(start_idx, end_idx))
        current_idx = end_idx
    
    return index


def build_flatl2_index(
    datastore_keys,
    use_gpu,
    knn_fp16,
):
    index = faiss.IndexFlatL2(datastore_keys.shape[1])

    if use_gpu:
        logger.info("Move index to gpu")
        resource, cloner_options = get_gpu_resource(knn_fp16)
        index = faiss.index_cpu_to_gpu(provider=resource, device=0, index=index, options=cloner_options)
    
    logger.info("Staring adding keys to the faiss index")
    index.add(datastore_keys.astype(np.float32))
    return index


def main(args: Namespace):
    if args.keys_dtype == "fp16":
        keys_dtype = np.float16
    else:
        keys_dtype = np.float32

    datastore_keys_path = os.path.join(args.datastore, "keys.npy")

    datastore_keys = np.memmap(
        datastore_keys_path, 
        dtype=keys_dtype, 
        mode="r", 
        shape=(args.datastore_size, args.keys_dimension)
    )
    
    if args.index_type == "IndexIVFPQ":
        index = build_ivfpq_index(
            datastore_keys,
            args.datastore,
            args.ncentroids,
            args.code_size,
            args.nbit_per_idx,
            args.nprobe,
            args.use_gpu,
            args.knn_fp16,
            args.num_trained_keys,
            args.batch_size,
            args.seed,
        )
    elif args.index_type == "IndexFlatL2":
        index = build_flatl2_index(
            datastore_keys,
            args.use_gpu,
            args.knn_fp16,
        )
    else:
        raise ValueError("Unknown index type")

    logger.info("Adding total {} keys".format(index.ntotal))

    final_index_path = os.path.join(args.datastore, "faiss.index")
    logger.info(f"Writing the final faiss index to {final_index_path}")
    if args.use_gpu:
        faiss.write_index(faiss.index_gpu_to_cpu(index), final_index_path)
    else:
        faiss.write_index(index, final_index_path)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--index-type", type=str, choices=["IndexFlatL2", "IndexIVFPQ"], default="IndexIVFPQ", help="faiss index type")

    parser.add_argument("--datastore", required=True, type=str, help="path to datastore directory")
    parser.add_argument("--datastore-size", required=True, type=int, help="the number of tokens in datastore")
    parser.add_argument("--keys-dimension", required=True, type=int, help="the feature dimension of datastore keys")
    parser.add_argument("--keys-dtype", default="fp16", choices=["fp16", "fp32"], type=str, help="keys dtype of the datastore")

    parser.add_argument("--use-gpu", action="store_true", default=False, help="whether to use GPU to build the faiss index")
    parser.add_argument("--knn-fp16", action="store_true", default=False, help="whether to perform intermediate calculations in float16")

    args, unknown = parser.parse_known_args()

    if args.index_type == "IndexIVFPQ":
        parser.add_argument("--seed", type=int, default=1, help="random seed for sampling the subset of keys to train the faiss index")

        parser.add_argument("--ncentroids", type=int, default=4096, help="number of centroids faiss should learn")
        parser.add_argument("--code-size", type=int, default=64, help="code size per vector in bytes")
        parser.add_argument("--nbit-per-idx", type=int, default=8, help="the number bits to index sub-codebook")

        parser.add_argument("--nprobe", type=int, default=32, help="number of clusters to query")

        parser.add_argument("--batch-size", default=500000, type=int, help="number of keys to add to the faiss index once time")
        parser.add_argument("--num-trained-keys", default=math.inf, type=int, help="maximum number of keys used for training the faiss index")
    return parser


def validate_args(args: Namespace):
    if args.knn_fp16:
        assert args.use_gpu, "knn_fp16 can only be performed on GPU"


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    validate_args(args)
    main(args)


if __name__ == "__main__":
    cli_main()
