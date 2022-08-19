import os
import math
import faiss
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from knn_models.data import PCKMTDataset


logger = logging.getLogger(__name__)


class PCATransform(nn.Module):
    def __init__(self, pca_input_size, pca_output_size, **kwargs):
        super().__init__()
        weight = torch.empty(pca_output_size, pca_input_size)
        bias = torch.empty((pca_output_size, ))
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
    
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


class CompactNet(nn.Module):
    def __init__(
        self, 
        compact_net_input_size, 
        compact_net_hidden_size,
        compact_net_output_size, 
        compact_net_dropout,
        **kwargs
    ):
        super().__init__()

        if compact_net_hidden_size is None:
            compact_net_hidden_size = compact_net_input_size // 4
        
        model = nn.Sequential(
            nn.Linear(compact_net_input_size, compact_net_hidden_size),
            nn.Tanh(),
            nn.Dropout(p=compact_net_dropout),
            nn.Linear(compact_net_hidden_size, compact_net_output_size)
        )
        
        # specific initialization
        nn.init.xavier_normal_(model[0].weight, gain=0.01)
        nn.init.xavier_normal_(model[-1].weight, gain=0.1)

        self.model = model

    def forward(self, input):
        return self.model(input)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def train_pckmt(
    datastore,
    datastore_size,
    keys_dimension,
    keys_dtype,
    reduced_keys_dimension,
    compact_net_hidden_size,
    compact_net_dropout,
    num_trained_keys,
    batch_size,
    num_workers,
    vocab_size,
    log_interval,
    max_update,
    max_epoch,
    update_freq,
    lr,
    betas,
    weight_decay,
    clip_norm,
    dbscan_eps,
    dbscan_min_samples,
    dbscan_max_samples,
    seed,
    transformed_datastore,
):
    set_seed(seed)

    compact_net = CompactNet(
        compact_net_input_size=keys_dimension,
        compact_net_hidden_size=compact_net_hidden_size,
        compact_net_output_size=reduced_keys_dimension,
        compact_net_dropout=compact_net_dropout
    )
    logger.info(compact_net)

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        compact_net = compact_net.cuda()

    optimizer = torch.optim.Adam(
        compact_net.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )

    train_dataset = PCKMTDataset(
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
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    def optimizer_step():
        # divide the grad by accumulated_bsz
        for p in compact_net.parameters():
            if p.grad is not None:
                p.grad.data.div_(accumulated_bsz)
        
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(compact_net.parameters(), clip_norm)
        
        optimizer.step()
        optimizer.zero_grad()

    num_updates = 0
    accumulated_bsz = 0
    should_stop = False

    max_update = max_update or math.inf

    target = torch.arange(0, batch_size, device="cuda")

    for epoch in range(1, max_epoch + 1):
        if should_stop:
            break

        train_dataset.set_epoch_and_rng(epoch)

        for i, sample in enumerate(train_dataloader):
            if should_stop:
                break

            if use_cuda:
                sample = utils.move_to_cuda(sample)
            
            # B x C
            positive_sample_key = sample["positive_sample_key"]
            pivot_sample_key = sample["pivot_sample_key"]
            negative_sample_key = sample["negative_sample_key"]

            # batch_size of the last batch may be not equal to batch_size
            bsz = positive_sample_key.size(0)

            input = torch.cat([positive_sample_key, pivot_sample_key, negative_sample_key], dim=0)
            output = compact_net(input)
            del input

            # B x Recuced_C
            positive_sample_key_proj, \
                pivot_sample_key_proj, \
                    negative_sample_key_proj = output.split(bsz)
            del output

            # B x 1 x Recuced_C
            positive_sample_key_proj = positive_sample_key_proj.unsqueeze(1)
            # 1 x B x Recuced_C
            pivot_sample_key_proj = pivot_sample_key_proj.unsqueeze(0)

            # B x 1 x Recuced_C * 1 x B x Recuced_C -> B x B
            positive_logits = (positive_sample_key_proj * pivot_sample_key_proj).sum(2)
            del positive_sample_key_proj

            # B x 1 x Recuced_C
            negative_sample_key_proj = negative_sample_key_proj.unsqueeze(1)

            negative_logits = (negative_sample_key_proj * pivot_sample_key_proj).sum(2)
            del negative_sample_key_proj, pivot_sample_key_proj

            # B x 2*B
            logits = torch.cat([positive_logits, negative_logits], dim=1)
            del positive_logits, negative_logits
            
            loss = F.cross_entropy(logits, target[: bsz])
            del logits

            loss.backward()

            accumulated_bsz += bsz

            if (i + 1) % update_freq == 0:
                optimizer_step()
                accumulated_bsz = 0
                num_updates += 1

                if num_updates % log_interval == 0:
                    logger.info(
                        f"update steps: {num_updates}, "
                        f"loss: {loss.item()}"
                    )

                if num_updates >= max_update:
                    should_stop = True
        
        if accumulated_bsz > 0:
            optimizer_step()
            accumulated_bsz = 0
            num_updates += 1
        
            if num_updates >= max_update:
                should_stop = True

        logger.info(f"Epoch {epoch} complete")

        checkpoint_name = f"checkpoint{epoch}.pt"
        checkpoint_path = os.path.join(transformed_datastore, checkpoint_name)
        torch.save(compact_net.state_dict(), checkpoint_path)

        logger.info(f"Saving {checkpoint_name} complete")
    
    if should_stop:
        checkpoint_name = "checkpoint_last.pt"
        checkpoint_path = os.path.join(transformed_datastore, checkpoint_name)
        torch.save(compact_net.state_dict(), checkpoint_path)
