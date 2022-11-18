import os
import faiss
import logging
import numpy as np

from sklearn.cluster import Birch


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

    logger.info("Start merging datastore keys")
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

    logger.info(f"Merging datastore keys complete. Pruned datastore size: {pruned_datastore_size}")
    
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

    logger.info("Start writing the pruned datastore to disk")
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

    logger.info("Start sampling indices")
    rng = np.random.default_rng(seed)
    sampeled_indices = rng.choice(np.arange(datastore_keys.shape[0]), size=pruned_datastore_size, replace=False)
    sampeled_indices = np.sort(sampeled_indices)

    logger.info(f"Sampling indices complete. Pruned datastore size: {pruned_datastore_size}")

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

    logger.info("Start writing the pruned datastore into disk")
    pruned_datastore_keys[:] = datastore_keys[sampeled_indices]
    pruned_datastore_values[:] = datastore_values[sampeled_indices]

    pruned_datastore_keys.flush()
    pruned_datastore_values.flush()

    logger.info("Random pruning complete")


def cluster_based_pruning(
    datastore,
    datastore_size,
    keys_dimension,
    keys_dtype,
    pruned_datastore,
    n_gram,
    translation_cost_threshold,
    sample_rate,
    minimum_sample_num,
    seed,
):
    assert 1 <= n_gram <= 4, \
        f"Does not implement pruning based on {n_gram}-gram yet!"

    datastore_4_gram_values_path = os.path.join(datastore, "4_gram_values.npy")
    datastore_4_gram_values = np.memmap(
        datastore_4_gram_values_path,
        dtype=np.int64,
        mode="r", 
        shape=(datastore_size, 4)
    )

    datastore_4_gram_values_probs_path = os.path.join(datastore, "4_gram_values_probs.npy")
    datastore_4_gram_values_probs = np.memmap(
        datastore_4_gram_values_probs_path,
        dtype=np.float32,
        mode="r", 
        shape=(datastore_size, 4)
    )

    non_padding_mask = datastore_4_gram_values != -1

    # datastore_size x 4
    log_4_gram_values_probs = -np.log(np.where(non_padding_mask, datastore_4_gram_values_probs, 1e-5))
    del non_padding_mask

    # datastore_size x n_gram
    log_4_gram_values_probs = log_4_gram_values_probs[:, :n_gram]

    # datastore_size x n_gram
    log_4_gram_values_ppl = np.empty_like(log_4_gram_values_probs)

    for n in range(1, n_gram + 1):
        log_4_gram_values_ppl[:, n - 1] = log_4_gram_values_probs[:, : n].sum(axis=1) / n

    del log_4_gram_values_probs

    # datastore_size
    translation_cost = np.min(log_4_gram_values_ppl, axis=1)
    del log_4_gram_values_ppl

    datastore_n_gram_values = datastore_4_gram_values[:, : n_gram]

    datastore_n_gram_values = np.where(datastore_n_gram_values == -1, 0, datastore_n_gram_values)
    hash_weight = np.array([0] + [np.exp(i + 1) for i in range(n_gram - 1)])
    hash_weight = np.expand_dims(hash_weight, axis=1)

    # n_gram: e.g., [30, 23, 40]
    # -> [30] and [23, 40]
    # -> [30] and [23 * exp(1) + 40 * exp(2) = 358.0827]
    # -> [30] and [358.0827 scaled to 0.3580827]
    # ->  30 + 0.3580827 = 30.3580827
    # the integer part is the final token vocab id
    datastore_n_gram_values_hash = np.matmul(datastore_n_gram_values, hash_weight).squeeze(1)
    datastore_n_gram_values_hash = datastore_n_gram_values_hash / \
            np.power(
                10, 
                np.log10(
                    np.where(
                        datastore_n_gram_values_hash == 0, 
                        1, 
                        datastore_n_gram_values_hash
                    )
                ).astype(np.int64) 
                + 1
            )
    
    datastore_n_gram_values_hash = datastore_n_gram_values[:, 0] + datastore_n_gram_values_hash
    del datastore_n_gram_values

    logger.info("Start collecting indice of each n-gram")
    # ngram to its indice
    ngram_to_idx = {}
    for idx, ngram in enumerate(datastore_n_gram_values_hash):
        ngram_to_idx.setdefault(ngram, []).append(idx)
    
    for ngram, idxs in ngram_to_idx.items():
        ngram_to_idx[ngram] = np.asarray(idxs, dtype=np.int64)

    rng = np.random.default_rng(seed)
    logger.info("Start cluster based pruning")

    for ngram, idxs in ngram_to_idx.items():
        ngram_translation_cost = translation_cost[idxs]

        sample_num = max(minimum_sample_num, int(idxs.shape[0] * sample_rate))

        if idxs.shape[0] <= sample_num:
            # do not prune it due to its sparseness
            continue

        if ngram_translation_cost.shape[0] <= 10000:
            # affinity greedy searching

            # size x size
            ngram_translation_cost_diff = \
                np.expand_dims(ngram_translation_cost, axis=0) - \
                    np.expand_dims(ngram_translation_cost, axis=1)

            cost_diff_below_threshold = np.abs(ngram_translation_cost_diff) <= translation_cost_threshold

            split_clusters = []

            remain_ngram_idx = np.arange(ngram_translation_cost.shape[0])

            # greedy clustering on translation costs
            while cost_diff_below_threshold.shape[0] > 0:
                num_cost_diff_below_threshold = cost_diff_below_threshold.sum(axis=1)
                row_idx = num_cost_diff_below_threshold.argmax()
                selected_ngram_mask = cost_diff_below_threshold[row_idx]
                split_clusters.append(remain_ngram_idx[selected_ngram_mask])

                non_selected_ngram_mask = np.logical_not(selected_ngram_mask)
                del selected_ngram_mask
                cost_diff_below_threshold = cost_diff_below_threshold[non_selected_ngram_mask]

                if non_selected_ngram_mask.any():
                    # to handle the case that cost_diff_below_threshold is an empty array
                    cost_diff_below_threshold = cost_diff_below_threshold[:, non_selected_ngram_mask]

                remain_ngram_idx = remain_ngram_idx[non_selected_ngram_mask]
                del non_selected_ngram_mask
            
            # uniform pruning
            for i, cluster_idx in enumerate(split_clusters):
                sample_num = max(min(minimum_sample_num, cluster_idx.shape[0]), int(cluster_idx.shape[0] * sample_rate))
                
                if sample_num >= cluster_idx.shape[0]:
                    continue
                
                split_clusters[i] = rng.choice(cluster_idx, sample_num, replace=False)

            remain_ngram_idx = np.concatenate(split_clusters, axis=0)
            del split_clusters
        else:
            # linear cluster (faster, not optical but acceptable)

            # greedy clustering on translation costs
            clustering = Birch(n_clusters=None, threshold=translation_cost_threshold)
            clustering.fit(np.expand_dims(ngram_translation_cost, axis=1))

            labels = clustering.labels_

            split_clusters = [list() for _ in range(labels.max() + 1)]

            # collect the isolated nodes
            isolated_nodes = []
            for i, label in enumerate(labels):
                # -1 means isolated sample
                if label == -1:
                    isolated_nodes.append(i)
                else:
                    split_clusters[label].append(i)
            
            # uniform pruning
            remain_ngram_idx = []
            for cluster_idx in split_clusters:
                if len(cluster_idx) == 0:
                    continue

                cluster_idx = np.asarray(cluster_idx)
                sample_num = max(min(minimum_sample_num, cluster_idx.shape[0]), int(cluster_idx.shape[0] * sample_rate))

                if sample_num < cluster_idx.shape[0]:
                    cluster_idx = rng.choice(cluster_idx, sample_num, replace=False)

                remain_ngram_idx.append(cluster_idx)
            
            # add the isolated nodes
            remain_ngram_idx.extend(np.asarray(isolated_nodes))
            del isolated_nodes
        
            remain_ngram_idx = np.concatenate(remain_ngram_idx, axis=0)
        
        ngram_to_idx[ngram] = idxs[remain_ngram_idx]
        del remain_ngram_idx
    
    remain_idx = []
    for idxs in ngram_to_idx.values():
        remain_idx.append(idxs)
    
    remain_idx = np.concatenate(remain_idx, axis=0)
    remain_idx = np.sort(remain_idx)
    logger.info(f"Datastore size after pruning: {remain_idx.shape[0]}")

    datastore_keys_path = os.path.join(datastore, "keys.npy")
    keys_dtype = np.float32 if keys_dtype == "fp32" else np.float16
    datastore_keys = np.memmap(
        datastore_keys_path, 
        dtype=keys_dtype, 
        mode="r", 
        shape=(datastore_size, keys_dimension)
    )

    pruned_datastore_keys_path = os.path.join(pruned_datastore, "keys.npy")
    logger.info(f"Saving pruned datastore keys in {pruned_datastore_keys_path}")
    pruned_datastore_keys = np.memmap(
        pruned_datastore_keys_path,
        dtype=keys_dtype,
        mode="w+",
        shape=(remain_idx.shape[0], keys_dimension)
    )
    pruned_datastore_keys[:] = datastore_keys[remain_idx]
    pruned_datastore_keys.flush()

    datastore_values_path = os.path.join(datastore, "values.npy")
    datastore_values = np.memmap(
        datastore_values_path,
        dtype=np.int64,
        mode="r",
        shape=(datastore_size, )
    )

    pruned_datastore_values_path = os.path.join(pruned_datastore, "values.npy")
    logger.info(f"Saving pruned datastore values in {pruned_datastore_values_path}")
    pruned_datastore_values = np.memmap(
        pruned_datastore_values_path,
        dtype=np.int64,
        mode="w+",
        shape=(remain_idx.shape[0], )
    )
    pruned_datastore_values[:] = datastore_values[remain_idx]
    pruned_datastore_values.flush()
    logger.info("Cluster based pruning complete")
