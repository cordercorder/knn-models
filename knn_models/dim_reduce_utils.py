import os
import re
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
        compact_net_dropout=0.0,
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
    keep_best_checkpoints,
    lr,
    betas,
    weight_decay,
    clip_norm,
    label_smoothing,
    dbscan_eps,
    dbscan_min_samples,
    dbscan_max_samples,
    seed,
    transformed_datastore,
):

    def save_checkpoint(state_dict, epoch, update, loss):
        if keep_best_checkpoints == 0:
            # do not save any checkpoint
            return 

        checkpoint_name = f"checkpoint.epoch_{epoch}.update_{update}.loss_{loss:.3f}.pt"
        checkpoint_path = os.path.join(transformed_datastore, checkpoint_name)

        if keep_best_checkpoints < 0:
            torch.save(state_dict, checkpoint_path)
        else:
            pattern = re.compile(r"checkpoint\.epoch_(\d+)\.update_(\d+)\.loss_(\d+.\d+)\.pt")

            checkpoints = []

            for file_name in os.listdir(transformed_datastore):
                match_obj = pattern.fullmatch(file_name)
                if match_obj is not None:
                    file_path = os.path.join(transformed_datastore, file_name)
                    ckpt_update = int(match_obj.group(2))
                    ckpt_loss = float(match_obj.group(3))
                    checkpoints.append((ckpt_loss, ckpt_update, file_path))
            
            checkpoints.sort()

            if len(checkpoints) + 1 > keep_best_checkpoints:
                if checkpoints[-1][0] >= loss:
                    # if loss of the new checkpoint is not greater than the old checkpoint
                    # remove the old checkpoint and save the new one
                    os.remove(checkpoints[-1][-1])
                    torch.save(state_dict, checkpoint_path)
            else:
                # if the number of saved checkpoint is not greater than keep_best_checkpoints-1
                # save the new checkpoint
                torch.save(state_dict, checkpoint_path)


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

        loss_per_epoch = 0.0
        n_correct_per_epoch = 0
        bsz_per_epoch = 0

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
            
            with torch.no_grad():
                predict = logits.argmax(1)
            
            n_correct = (predict == target[: bsz]).sum().item()
            del predict

            n_correct_per_epoch += n_correct

            loss = F.cross_entropy(
                input=logits, 
                target=target[: bsz], 
                reduction="sum", 
                label_smoothing=label_smoothing
            )

            del logits

            loss.backward()

            accumulated_bsz += bsz

            loss_per_epoch += loss.item()
            bsz_per_epoch += bsz

            if (i + 1) % update_freq == 0:
                optimizer_step()
                accumulated_bsz = 0
                num_updates += 1

                if num_updates % log_interval == 0:
                    logger.info(
                        f"Epoch {epoch}, update steps: {num_updates}, "
                        f"loss: {loss_per_epoch / bsz_per_epoch: .5f}, "
                        f"acc: {n_correct_per_epoch / bsz_per_epoch: .5f}"
                    )

                if num_updates >= max_update:
                    should_stop = True
        
        if accumulated_bsz > 0:
            optimizer_step()
            accumulated_bsz = 0
            num_updates += 1
        
            if num_updates >= max_update:
                should_stop = True
        
        epoch_loss = loss_per_epoch / bsz_per_epoch
        epoch_acc = n_correct_per_epoch / bsz_per_epoch
        logger.info(f"Epoch loss: {epoch_loss: .5f}, Epoch acc: {epoch_acc: .5f}")

        save_checkpoint(compact_net.state_dict(), epoch, num_updates, epoch_loss)
    
    logger.info("Training PCKMT complete")


def apply_pckmt(
    datastore,
    datastore_size,
    keys_dimension,
    keys_dtype,
    reduced_keys_dimension,
    compact_net_hidden_size,
    batch_size,
    log_interval,
    checkpoint_name,
    transformed_datastore,
):
    datastore_keys_path = os.path.join(datastore, "keys.npy")
    logger.info(f"Loading {datastore_size} datastore keys from {datastore_keys_path}")
    keys_dtype = np.float32 if keys_dtype == "fp32" else np.float16
    datastore_keys = np.memmap(
        datastore_keys_path, 
        dtype=keys_dtype, 
        mode="r", 
        shape=(datastore_size, keys_dimension)
    )

    reduced_datastore_keys_path = os.path.join(transformed_datastore, "keys.npy")
    logger.info(f"Create reduced datastore keys in {reduced_datastore_keys_path}")
    recuced_datastore_keys = np.memmap(
        reduced_datastore_keys_path,
        dtype=keys_dtype,
        mode="w+",
        shape=(datastore_size, reduced_keys_dimension)
    )

    compact_net = CompactNet(
        compact_net_input_size=keys_dimension,
        compact_net_hidden_size=compact_net_hidden_size,
        compact_net_output_size=reduced_keys_dimension,
    )

    checkpoint_path = os.path.join(transformed_datastore, checkpoint_name)
    compact_net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    logger.info(compact_net)

    use_cuda = torch.cuda.is_available()
    
    if use_cuda:
        compact_net = compact_net.cuda()
    
    compact_net.eval()
    
    num_batches = datastore_keys.shape[0] // batch_size + \
        int(datastore_keys.shape[0] % batch_size != 0)

    current_idx = 0
    for i in range(num_batches):
        start_idx = current_idx
        end_idx = min(start_idx + batch_size, datastore_keys.shape[0])

        sample_keys = datastore_keys[start_idx: end_idx].astype(np.float32)
        sample_keys = torch.from_numpy(sample_keys)

        if use_cuda:
            sample_keys = sample_keys.cuda()

        with torch.no_grad():
            reduced_sample_keys = compact_net(sample_keys)
            del sample_keys
        
        reduced_sample_keys = reduced_sample_keys.cpu().numpy().astype(keys_dtype)
        recuced_datastore_keys[start_idx: end_idx] = reduced_sample_keys

        current_idx = end_idx
    
        if (i + 1) % log_interval == 0:
            logger.info(f"{i + 1} / {num_batches} batches complete")
    
    logger.info("Applying PCKMT complete")
