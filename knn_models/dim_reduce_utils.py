import os
import faiss
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class PCATransform(nn.Module):
    def __init__(self, pca_input_dim, pca_output_dim, **kwargs):
        super().__init__()
        weight = torch.empty(pca_output_dim, pca_input_dim)
        bias = torch.empty((pca_output_dim, ))
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
    
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


def pca_dimension_reduction(
    datastore,
    datastore_size,
    keys_dimension,
    keys_dtype,
    reduced_keys_dimension,
    random_rotation,
    transformed_datastore,
):
    datastore_keys_path = os.path.join(datastore, "keys.npy")
    keys_dtype = np.float32 if keys_dtype == "fp32" else np.float16
    datastore_keys = np.memmap(
        datastore_keys_path, 
        dtype=keys_dtype, 
        mode="r", 
        shape=(datastore_size, keys_dimension)
    )

    pca = faiss.PCAMatrix(keys_dimension, reduced_keys_dimension, 0, random_rotation)

    logger.info("Start converting the data type of datastore keys to 32-bits float")
    datastore_keys_float32 = datastore_keys.astype(np.float32)

    logger.info("Start PCA training")
    pca.train(datastore_keys_float32)

    logger.info("Start applying PCA")
    transformed_datastore_keys_float32 = pca.apply_py(datastore_keys_float32)
    transformed_datastore_keys_path = os.path.join(transformed_datastore, "keys.npy")

    transformed_datastore_keys = np.memmap(
        transformed_datastore_keys_path,
        dtype=keys_dtype, 
        mode="w+",
        shape=(datastore_size, reduced_keys_dimension)
    )

    logger.info("Start writing transformed datastore keys into disk")
    transformed_datastore_keys[:] = transformed_datastore_keys_float32.astype(keys_dtype)
    transformed_datastore_keys.flush()

    logger.info("Start saving PCATransform")
    pca_transform = PCATransform(keys_dimension, reduced_keys_dimension)

    # copy the weight and bias used for linear transformation in PCA to PCATransform module
    pca_transform.weight.copy_(
        torch.from_numpy(
            faiss.vector_to_array(pca.A).reshape(reduced_keys_dimension, keys_dimension)
        )
    )
    pca_transform.bias.copy_(torch.from_numpy(faiss.vector_to_array(pca.b)))

    pca_transform_path = os.path.join(transformed_datastore, "transform.pt")
    torch.save(pca_transform.state_dict(), pca_transform_path)

    logger.info("PCA dimension reduction complete")
