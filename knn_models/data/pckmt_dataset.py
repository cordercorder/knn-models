import os
import torch
import logging
import numpy as np

from sklearn.cluster import DBSCAN
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class PCKMTDataset(Dataset):
    def __init__(
        self, 
        datastore, 
        datastore_size, 
        keys_dimension, 
        keys_dtype,
        num_trained_keys,
        vocab_size,
        dbscan_eps,
        dbscan_min_samples,
        dbscan_max_samples,
        seed,
    ):
        super().__init__()

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

        if num_trained_keys < datastore_keys.shape[0]:
            logger.info("Start sampling indices")
            sampeled_indices = rng.choice(np.arange(datastore_keys.shape[0]), size=num_trained_keys, replace=False)
            sampeled_indices = np.sort(sampeled_indices)

            logger.info("Start fetching datastore keys and values specified by sampled indices")
            datastore_keys = datastore_keys[sampeled_indices]
            datastore_values = datastore_values[sampeled_indices]

        logger.info("Start collecting datastore keys for each value")
        keys_per_value = [list() for _ in range(vocab_size)]
        for i in range(datastore_keys.shape[0]):
            # value is a scalar
            value = datastore_values[i]

            if value == 3:
                # 3 is UNK
                continue

            # key is a numpy array
            key = datastore_keys[i]
            keys_per_value[value].append(key)
        
        logger.info("Start clustering datastore keys")

        clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)

        datastore_keys_new = []

        for value, keys in enumerate(keys_per_value):

            # keys is a list of numpy array
            num_keys = len(keys)

            if num_keys == 0:
                continue

            if num_keys <= dbscan_min_samples:
                keys = np.asarray(keys, dtype=np.float32)
                datastore_keys_new.append(keys)
                continue
            
            if num_keys > dbscan_max_samples:
                keys = rng.choice(keys, size=dbscan_max_samples, replace=False)
            
            keys = np.asarray(keys, dtype=np.float32)
            clustering.fit(keys)

            labels = clustering.labels_

            # [[key1, key2, key3], [key4, ...], ...]
            keys_per_cluster = [list() for _ in range(labels.max() + 1)]
            for i in range(labels.shape[0]):
                label = labels[i]

                # -1 means noisy sample
                if label != -1:
                    keys_per_cluster[label].append(keys[i])
            
            for keys in keys_per_cluster:
                if len(keys) > 0:
                    # [ndarray1, ndarray2, ...] -> ndarray
                    keys = np.asarray(keys)
                    datastore_keys_new.append(keys)

        logger.info(f"Cluster number: {len(datastore_keys_new)}")

        centroid_keys = []

        num_clusters = len(datastore_keys_new)
        for i in range(num_clusters):
            _centroid_key = datastore_keys_new[i].mean(axis=0)
            centroid_keys.append(_centroid_key)
        
        # indices of cluster which contains more than one keys
        cluster_idx_gt_one = [i for i, keys in enumerate(datastore_keys_new) if keys.shape[0] > 1]

        # set the attribute
        self.seed = seed
        self.datastore_keys_new = datastore_keys_new

        self.centroid_keys = centroid_keys
        self.cluster_idx_gt_one = cluster_idx_gt_one
    
    def set_epoch_and_rng(self, epoch):
        self.epoch = epoch
        self.rng = np.random.default_rng(self.seed + epoch)

    def __getitem__(self, index):
        index = self.cluster_idx_gt_one[index]

        pivot_sample_key = self.centroid_keys[index]

        positive_sample_key = self.rng.choice(self.datastore_keys_new[index], size=1).squeeze(0)

        while True:
            negative_sample_cluster_id = self.rng.choice(len(self.datastore_keys_new), size=1)[0]
            if negative_sample_cluster_id != index:
                break
        
        negative_sample_key = self.rng.choice(self.datastore_keys_new[negative_sample_cluster_id], size=1).squeeze(0)

        return {
            "pivot_sample_key": torch.from_numpy(pivot_sample_key),
            "positive_sample_key": torch.from_numpy(positive_sample_key),
            "negative_sample_key": torch.from_numpy(negative_sample_key),
        }

    def __len__(self):
        return len(self.cluster_idx_gt_one)
