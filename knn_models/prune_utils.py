import os
import faiss
import logging
import numpy as np


logger = logging.getLogger(__name__)


def save_datastore_knn(
    datastore_keys, 
    index_path, 
    use_gpu, 
    knn_fp16, 
    nprobe,
    num_neighbors, 
    batch_size,
    save_knn_distance,
    log_interval,
):
    """"Retrieve k-nearest neighbors for each keys in datastore"""
    index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)

    if use_gpu:
        logger.info("Moving index to GPU")
        resource = faiss.StandardGpuResources()
        cloner_options = None
        if knn_fp16:
            cloner_options = faiss.GpuClonerOptions()
            cloner_options.useFloat16 = True
        
        index = faiss.index_cpu_to_gpu(provider=resource, device=0, index=index, options=cloner_options)

    logger.info(f"Setting nprobe of index to {nprobe}")
    index.nprobe = nprobe

    datastore = os.path.dirname(datastore_keys.filename)
    datastore_knn_idx_path = os.path.join(datastore, "datastore_knn_idx.npy")
    datastore_knn_idx = np.memmap(
        datastore_knn_idx_path,
        dtype=np.int64,
        mode="w+",
        shape=(datastore_keys.shape[0], num_neighbors)
    )

    if save_knn_distance:
        datastore_knn_distence_path = os.path.join(datastore, "datastore_knn_distance.npy")
        datastore_knn_distance = np.memmap(
            datastore_knn_distence_path,
            dtype=np.float32,
            mode="w+",
            shape=(datastore_keys.shape[0], num_neighbors)
        )
    else:
        datastore_knn_distance = None

    current_idx = 0
    num_batches = datastore_keys.shape[0] // batch_size + int(datastore_keys.shape[0] % batch_size != 0)

    for batch_id in range(num_batches):
        start_idx = current_idx
        end_idx = min(start_idx + batch_size, datastore_keys.shape[0])

        # B x k
        distance, idx = index.search(datastore_keys[start_idx: end_idx].astype(np.float32), num_neighbors)
        datastore_knn_idx[start_idx: end_idx] = idx

        if datastore_knn_distance is not None:
            datastore_knn_distance[start_idx: end_idx] = distance
        
        current_idx = end_idx

        if (batch_id + 1) % log_interval == 0:
            logger.info(f"{batch_id + 1} / {num_batches} batches have been processed!")
    
    datastore_knn_idx.flush()
    if datastore_knn_distance is not None:
        datastore_knn_distance.flush()
    
    logger.info("Saving datastore knn complete")


def greedy_merge_pruning(
    datastore,
    datastore_size,
    keys_dimension,
    keys_dtype,
    pruned_datastore,
    use_gpu,
    batch_size,
    num_neighbors,
    nprobe,
    knn_fp16,
    save_knn_distance,
    log_interval,
    seed,
):
    datastore_keys_path = os.path.join(datastore, "keys.npy")
    keys_dtype = np.float32 if keys_dtype == "fp32" else np.float16
    datastore_keys = np.memmap(
        datastore_keys_path, 
        dtype=keys_dtype, 
        mode="r", 
        shape=(datastore_size, keys_dimension)
    )

    save_datastore_knn(
        datastore_keys,
        os.path.join(datastore, "faiss.index"),
        use_gpu,
        knn_fp16,
        nprobe,
        num_neighbors,
        batch_size,
        save_knn_distance,
        log_interval
    )

    datastore_weight = np.ones((datastore_size, ), dtype=np.int64)

    datastore_knn_idx_path = os.path.join(datastore, "datastore_knn_idx.npy")
    datastore_knn_idx = np.memmap(
        datastore_knn_idx_path,
        dtype=np.int64,
        mode="r",
        shape=(datastore_size, num_neighbors)
    )

    datastore_values_path = os.path.join(datastore, "values.npy")
    datastore_values = np.memmap(
        datastore_values_path,
        dtype=np.int64,
        mode="r",
        shape=(datastore_size, )
    )

    rng = np.random.default_rng(seed)

    for i in rng.permutation(datastore_size):
        for neighbor_idx in datastore_knn_idx[i]:
            if neighbor_idx != i and \
                datastore_weight[neighbor_idx] == 1 and \
                    datastore_values[i] == datastore_values[neighbor_idx]:
                
                datastore_weight[neighbor_idx] = 0
                datastore_weight[i] += 1
    
    pruned_datastore_weight_mask = datastore_weight > 0
    pruned_datastore_size = pruned_datastore_weight_mask.sum()

    logger.info(f"Pruned datastore size: {pruned_datastore_size}")
    
    pruned_datastore_keys_path = os.path.join(pruned_datastore, "keys.npy")
    pruned_datastore_keys = np.memmap(
        pruned_datastore_keys_path,
        dtype=keys_dtype,
        mode="w+",
        shape=(pruned_datastore_size, keys_dimension)
    )

    pruned_datastore_values_path = os.path.join(pruned_datastore, "values.npy")
    pruned_datastore_values = np.memmap(
        pruned_datastore_values_path,
        dtype=np.int64,
        mode="w+",
        shape=(pruned_datastore_size, )
    )

    pruned_datastore_weight_path = os.path.join(pruned_datastore, "weight.npy")
    pruned_datastore_weight = np.memmap(
        pruned_datastore_weight_path,
        dtype=np.float32,
        mode="w+",
        shape=(pruned_datastore_size, )
    )

    pruned_datastore_weight_indices = np.nonzero(pruned_datastore_weight_mask)
    pruned_datastore_keys[:] = datastore_keys[pruned_datastore_weight_indices]
    pruned_datastore_values[:] = datastore_values[pruned_datastore_weight_indices]
    pruned_datastore_weight[:] = datastore_weight[pruned_datastore_weight_indices].astype(np.float32)
    
    pruned_datastore_keys.flush()
    pruned_datastore_values.flush()
    pruned_datastore_weight.flush()

    logger.info("Greedy merge pruning complete")


def random_pruning(
    datastore,
    datastore_size,
    keys_dimension,
    keys_dtype,
    pruned_datastore,
    pruned_datastore_size,
    seed,
):
    datastore_keys_path = os.path.join(datastore, "keys.npy")
    keys_dtype = np.float32 if keys_dtype == "fp32" else np.float16
    datastore_keys = np.memmap(
        datastore_keys_path, 
        dtype=keys_dtype, 
        mode="r", 
        shape=(datastore_size, keys_dimension)
    )

    datastore_values_path = os.path.join(datastore, "values.npy")
    datastore_values = np.memmap(
        datastore_values_path,
        dtype=np.int64,
        mode="r",
        shape=(datastore_size, )
    )

    rng = np.random.default_rng(seed)
    sampeled_indices = rng.choice(np.arange(datastore_keys.shape[0]), size=pruned_datastore_size, replace=False)
    sampeled_indices = np.sort(sampeled_indices)

    logger.info(f"Pruned datastore size: {pruned_datastore_size}")

    pruned_datastore_keys_path = os.path.join(pruned_datastore, "keys.npy")
    pruned_datastore_keys = np.memmap(
        pruned_datastore_keys_path,
        dtype=keys_dtype,
        mode="w+",
        shape=(pruned_datastore_size, keys_dimension)
    )

    pruned_datastore_values_path = os.path.join(pruned_datastore, "values.npy")
    pruned_datastore_values = np.memmap(
        pruned_datastore_values_path,
        dtype=np.int64,
        mode="w+",
        shape=(pruned_datastore_size, )
    )

    pruned_datastore_keys[:] = datastore_keys[sampeled_indices]
    pruned_datastore_values[:] = datastore_values[sampeled_indices]

    pruned_datastore_keys.flush()
    pruned_datastore_values.flush()

    logger.info("Random pruning complete")
