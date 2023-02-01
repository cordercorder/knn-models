import os
import re
import math
import faiss
import torch
import logging
import numpy as np

from torch import Tensor
from fairseq import utils
from knn_models.dataclass import (
    KnnConfig, 
    RobustKnnConfig,
    AdaptiveKnnConfig,
    KernelSmoothedKnnConfig,
)
from typing import Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


class KnnSearch:
    def __init__(self, cfg: KnnConfig):
        self.cfg = cfg

        if not cfg.saving_mode:
            self.setup_knn_search()

    def setup_knn_search(self):
        index_path = os.path.join(self.cfg.datastore, "faiss.index")
        index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)

        if min(self.cfg.knn_device_id) >= 0:
            logger.info(f"Moving index to GPU {self.cfg.knn_device_id}")

            if len(self.cfg.knn_device_id) == 1:
                resource = faiss.StandardGpuResources()
                cloner_options = None
                if self.cfg.knn_fp16:
                    cloner_options = faiss.GpuClonerOptions()
                    cloner_options.useFloat16 = True
                
                index = faiss.index_cpu_to_gpu(
                    provider=resource, 
                    device=self.cfg.knn_device_id[0], 
                    index=index, 
                    options=cloner_options
                )
            else:
                # when multiple GPU devices are specified, shard the index across GPUs
                # this can be useful if the datastore is too large to fit into one GPU device
                gpu_resources = [faiss.StandardGpuResources() for _ in range(len(self.cfg.knn_device_id))]
                
                resource_vector = faiss.GpuResourcesVector()
                device_vector = faiss.IntVector()
                for i, idx in enumerate(self.cfg.knn_device_id):
                    resource_vector.push_back(gpu_resources[i])
                    device_vector.push_back(idx)
                
                cloner_options = faiss.GpuMultipleClonerOptions()
                cloner_options.shard = True

                if self.cfg.knn_fp16:
                    cloner_options.useFloat16 = True

                index = faiss.index_cpu_to_gpu_multiple(
                    provider=resource_vector, 
                    devices=device_vector, 
                    index=index, 
                    options=cloner_options
                )

        else:
            assert len(self.cfg.knn_device_id) == 1, "Only one device can be used when perform kNN search on CPU"

        if hasattr(index, "nprobe"):
            logger.info(f"Setting nprobe of index to {self.cfg.nprobe}")
            index.nprobe = self.cfg.nprobe

        datastore_values_path = os.path.join(self.cfg.datastore, "values.npy")
        logger.info(f"Loading datastore values from {datastore_values_path}")
        datastore_values = np.memmap(
            datastore_values_path,
            dtype=np.int64,
            mode="r",
            shape=(self.cfg.datastore_size, )
        )

        if self.cfg.load_keys:
            datastore_keys_path = os.path.join(self.cfg.datastore, "keys.npy")
            keys_dtype = np.float32 if self.cfg.keys_dtype == "fp32" else np.float16
            logger.info(f"Loading datastore keys from {datastore_keys_path}")
            datastore_keys = np.memmap(
                datastore_keys_path, 
                dtype=keys_dtype, 
                mode="r", 
                shape=(self.cfg.datastore_size, self.cfg.keys_dimension)
            )
        
        if self.cfg.load_value_weights:
            datastore_value_weights_path = os.path.join(self.cfg.datastore, "weight.npy")
            logger.info(f"Loading the weight of datastore values from {datastore_value_weights_path}")
            datastore_value_weights = np.memmap(
                datastore_value_weights_path,
                dtype=np.float32,
                mode="r",
                shape=(self.cfg.datastore_size, )
            )
            
        if self.cfg.move_to_memory:
            logger.info("Moving datastore values into CPU memory")
            _datastore_values = np.empty((self.cfg.datastore_size, ), dtype=np.int64)
            _datastore_values[:] = datastore_values
            datastore_values = _datastore_values

            if self.cfg.load_keys:
                logger.info("Moving datastore keys into CPU memory")
                _datastore_keys = np.empty((self.cfg.datastore_size, self.cfg.keys_dimension), dtype=keys_dtype)
                _datastore_keys[:] = datastore_keys
                datastore_keys = _datastore_keys
            
            if self.cfg.load_value_weights:
                logger.info("Moving the weight of datastore values into CPU memory")
                _datastore_value_weights = np.empty((self.cfg.datastore_size, ), dtype=np.float32)
                _datastore_value_weights[:] = datastore_value_weights
                datastore_value_weights = _datastore_value_weights
        
        self.index = index
        self.datastore_values = datastore_values

        self.datastore_keys = datastore_keys if self.cfg.load_keys else None
        
        self.datastore_value_weights = datastore_value_weights if self.cfg.load_value_weights else None

    def retrieve(self, queries):
        bsz, seq_len = queries.size()[:2]
        # B*T x C
        queries = queries.contiguous().view(-1, queries.size(-1))

        queries_device = queries.device

        # B*T x K
        distance, idx = self.index.search(queries.cpu().float().numpy(), self.cfg.num_neighbors)
        del queries

        # distance and queries should have the same device
        distance = torch.from_numpy(distance).to(queries_device)
        distance = distance.view(bsz, seq_len, -1)

        distance.neg_()

        if self.datastore_value_weights is not None:
            weight = torch.from_numpy(self.datastore_value_weights[idx]).to(queries_device)
            weight = weight.view(bsz, seq_len, -1)
            # following the original implementation in `Efficient Nearest Neighbor Language Models`
            distance.add_(weight.log_())
            del weight
        
        distance.div_(self.cfg.temperature_value)
        
        distance = utils.softmax(distance, dim=-1)
        
        distance.mul_(self.cfg.lambda_value)

        tgt_idx = torch.from_numpy(self.datastore_values[idx]).to(queries_device)
        tgt_idx = tgt_idx.view(bsz, seq_len, -1)
        return {"knn_prob": distance, "tgt_idx": tgt_idx, "lambda_value": self.cfg.lambda_value}


class AdaptiveKnnSearch(KnnSearch):
    def __init__(self, cfg: AdaptiveKnnConfig):
        super().__init__(cfg)

        self.value_count_mask = self.get_value_count_mask(self.cfg.num_neighbors)
        self.distance_mask = self.get_distance_mask(self.cfg.num_neighbors)
        self.reduced_k = self.distance_mask.size(0)

        self.meta_k_network = None
    
    @staticmethod
    def get_value_count_mask(length):
        # 0 1 1
        # 0 0 1
        # 0 0 0
        value_count_mask = torch.full([length, length], True, dtype=torch.bool)
        value_count_mask.triu_(diagonal=1)
        return value_count_mask
    
    @staticmethod
    def get_distance_mask(length):
        # 0 1 1
        # 0 0 1
        # 0 0 0
        distance_mask = torch.full([length, length], True, dtype=torch.bool)
        distance_mask.triu_(diagonal=1)

        selected_index = []
        idx = 1
        for _ in range(int(math.log2(length)) + 1):
            selected_index.append(idx - 1)
            idx *= 2

        return distance_mask[selected_index]
    
    def retrieve(self, queries):
        bsz, seq_len = queries.size()[:2]
        # B*T x C
        queries = queries.contiguous().view(-1, queries.size(-1))

        queries_device = queries.device
        queries_dtype = queries.dtype

        # B*T x K
        distance, idx = self.index.search(queries.cpu().float().numpy(), self.cfg.num_neighbors)
        del queries

        tgt_idx = torch.from_numpy(self.datastore_values[idx]).to(queries_device)
        tgt_idx = tgt_idx.view(bsz, seq_len, -1)

        # distance and queries should have the same device
        distance = torch.from_numpy(distance).to(queries_device)
        distance = distance.view(bsz, seq_len, -1)

        value_count = self.get_value_count(tgt_idx)
        meta_k_network_input = torch.cat([distance, value_count.float()], dim=2)
        del value_count

        # in case of mix precision training
        meta_k_network_input = meta_k_network_input.type(queries_dtype)

        # B x T x (R_k+1)
        p_meta = self.meta_k_network(meta_k_network_input)
        p_meta = utils.softmax(p_meta, dim=2)
        del meta_k_network_input

        # B x T x 1
        lambda_value = 1.0 - p_meta[: , :, :1]

        # B x T x R_k
        p_meta = p_meta[:, :, 1:]

        distance.neg_().div_(self.cfg.temperature_value)

        # B x T x R_k x K
        distance = distance.unsqueeze_(2).expand(bsz, seq_len, self.reduced_k, self.cfg.num_neighbors)
        self.distance_mask = self.distance_mask.to(distance.device)

        distance = distance.masked_fill(self.distance_mask, float("-inf"))

        distance = utils.softmax(distance, dim=-1)

        # B x T x K
        distance = p_meta.unsqueeze(2).matmul(distance).squeeze(2)

        return {"knn_prob": distance, "tgt_idx": tgt_idx, "lambda_value": lambda_value}

    def get_value_count(self, tgt_idx, relative_label_count=False):
        bsz, seq_len, k = tgt_idx.size()

        # B x T x K x K
        tgt_idx = tgt_idx.unsqueeze(2).expand(bsz, seq_len, k, k)

        self.value_count_mask = self.value_count_mask.to(tgt_idx.device)
        tgt_idx = tgt_idx.masked_fill(self.value_count_mask, -1)

        value_sorted = tgt_idx.sort(dim=3)[0]

        value_sorted[:, :, :, 1:].mul_(
            (value_sorted[:, :, :, 1:] - value_sorted[:, :, :, :-1]).ne_(0).long()
        )

        # B x T x K
        value_count = value_sorted.ne_(0).long().sum(dim=3)
        value_count[:, :, :-1].sub_(1)

        if relative_label_count:
            value_count[:, :, 1:] = value_count[:, :, 1:] - value_count[:, :, :-1]
        
        return value_count


class RobustKnnSearch(AdaptiveKnnSearch):
    def __init__(self, cfg: RobustKnnConfig):
        # skip __init__ in AdaptiveKnnSearch
        super(AdaptiveKnnSearch, self).__init__(cfg)
        self.value_count_mask = self.get_value_count_mask(self.cfg.num_neighbors)

        self.W_1_5 = None
        self.W_2 = None
        self.W_3 = None
        self.W_4 = None
        self.W_6 = None

        self.output_projection = None

        assert self.cfg.load_keys, "please set `--load-keys` flag when using RobustKnnSearch"
    
    def retrieve(self, queries, mt_prob, target, num_updates, training):
        bsz, seq_len = queries.size()[:2]
        # B*T x C
        queries = queries.contiguous().view(-1, queries.size(-1))

        queries_device = queries.device
        queries_dtype = queries.dtype

        queries = queries.float()

        # B*T x K
        idx = self.index.search(queries.cpu().numpy(), self.cfg.num_neighbors)[1]
        # B*T x K x C
        retrieved_keys = torch.from_numpy(self.datastore_keys[idx]).to(queries_device)

        # recompute distance and idx
        # B*T x K
        distance = (retrieved_keys.float() - queries.unsqueeze(1)).pow(2).sum(dim=2)
        del retrieved_keys
        distance, indices = torch.sort(distance, dim=1)
        indices = indices.cpu().numpy()
        idx = np.take_along_axis(idx, indices=indices, axis=1)
        del indices

        # B*T x K x C
        retrieved_keys = torch.from_numpy(self.datastore_keys[idx]).to(queries_device)
        # B x T x K x C
        retrieved_keys = retrieved_keys.view(bsz, seq_len, self.cfg.num_neighbors, -1).float()

        tgt_idx = torch.from_numpy(self.datastore_values[idx]).to(queries_device)
        del idx

        tgt_idx = tgt_idx.view(bsz, seq_len, -1)

        if training:
            del distance

            alpha = self.cfg.alpha_zero * math.exp(- num_updates / self.cfg.beta)

            # as the random numbers generated by `torch.rand([bsz, seq_len]).to(queries_device)` and `torch.rand([bsz, seq_len], device=queries_device)` 
            # are different, we use `torch.rand([bsz, seq_len]).to(queries_device)` below to keep the same implementation with official
            # B x T
            random_alpha = torch.rand([bsz, seq_len]).to(queries_device)

            # the index of padding token is 1
            padding_mask = target.ne(1)

            # B x T
            perturbation_mask = (random_alpha < alpha) & padding_mask
            del random_alpha

            # perturbation for the first problem as shown in Figure 4(a) of the paper
            # B x T x K x C
            perturbation = torch.randn_like(retrieved_keys) * 0.01
            perturbation.masked_fill_(
                perturbation_mask.logical_not_().unsqueeze_(2).unsqueeze_(3), 
                0.0
            )
            del perturbation_mask

            retrieved_keys.add_(perturbation)

            # perturbation for the second problem as shown in Figure 4(b) of the paper
            # B x T x C
            queries = queries.view(bsz, seq_len, -1)
            perturbation = torch.randn_like(queries) * 0.01
            pseudo_keys = queries + perturbation
            del perturbation

            # B x T x K x C
            pseudo_keys = torch.cat([pseudo_keys.unsqueeze(2), retrieved_keys[:, :, :-1]], dim=2)

            # B x T x K
            target = target.unsqueeze(2)
            pseudo_tgt_idx = torch.cat([target, tgt_idx[:, :, :-1]], dim=2)

            # B x T
            pseudo_keys_mask = (tgt_idx == target).any(dim=2)

            # equivalent to: ~((tgt_idx == target).any(dim=2)) & padding_mask
            pseudo_keys_mask.logical_not_().logical_and_(padding_mask)
            del padding_mask

            # as the random numbers generated by `torch.rand([bsz, seq_len]).to(queries_device)` and `torch.rand([bsz, seq_len], device=queries_device)` 
            # are different, we use `torch.rand([bsz, seq_len]).to(queries_device)` below to keep the same implementation with official
            random_alpha = torch.rand([bsz, seq_len]).to(queries_device)
            # equivalent to: pseudo_keys_mask = pseudo_keys_mask & (random_alpha < alpha)
            pseudo_keys_mask.logical_and_(random_alpha < alpha)
            del random_alpha

            # B x T x 1
            pseudo_keys_mask.unsqueeze_(2)

            pseudo_tgt_idx = torch.where(pseudo_keys_mask, pseudo_tgt_idx, tgt_idx)
            tgt_idx = pseudo_tgt_idx

            pseudo_keys = torch.where(pseudo_keys_mask.unsqueeze_(3), pseudo_keys, retrieved_keys)
            del pseudo_keys_mask

            retrieved_keys = pseudo_keys
            del pseudo_keys

            # B x T x K
            distance = (retrieved_keys - queries.unsqueeze(2)).pow(2).sum(dim=3)
            del queries

            distance, indices = torch.sort(distance, dim=2)
            tgt_idx = tgt_idx.gather(dim=2, index=indices)
        else:
            distance = distance.view(bsz, seq_len, -1)
        
        # in case of mix precision trianing
        distance = distance.type(queries_dtype)
        retrieved_keys = retrieved_keys.type(queries_dtype)

        # B x T x K
        value_count = self.get_value_count(tgt_idx)
        value_count = value_count.type(queries_dtype)

        # B x T x 2
        temparature_s_knn = self.W_1_5(
            torch.tanh(
                self.W_2(
                    torch.cat([distance, value_count], dim=2)
                )
            )
        )
        del value_count

        # B x T x 1
        s_knn, temparature = temparature_s_knn.chunk(2, dim=2)
        del temparature_s_knn

        # B x T x K x V
        keys_prob = utils.log_softmax(self.output_projection(retrieved_keys), dim=3)
        del retrieved_keys

        if training:
            # B x T x K
            keys_prob = keys_prob.gather(dim=3, index=pseudo_tgt_idx.unsqueeze_(3)).squeeze(3)
            del pseudo_tgt_idx
            keys_prob = keys_prob.gather(dim=2, index=indices)
            del indices
        else:
            # B x T x K
            keys_prob = keys_prob.gather(dim=3, index=tgt_idx.unsqueeze(3)).squeeze(3)
        
        keys_prob = keys_prob.type(queries_dtype)

        # B x T x V
        mt_prob = mt_prob.log()
        mt_prob = mt_prob.type(queries_dtype)
        
        # B x T x K
        mt_prob_of_tgt_idx = mt_prob.gather(dim=2, index=tgt_idx)
        # B x T x K x 1
        c_k = self.W_3(
            torch.tanh(
                self.W_4(
                    torch.stack([keys_prob, mt_prob_of_tgt_idx], dim=3)
                )
            )
        )
        # B x T x K
        c_k = c_k.squeeze(3)

        # B x T x 8
        top_mt_prob = mt_prob.topk(8, dim=2)[0]
        del mt_prob

        # B x T x 1
        s_nmt = self.W_6(torch.cat([top_mt_prob, keys_prob, mt_prob_of_tgt_idx], dim=2))
        del top_mt_prob, keys_prob, mt_prob_of_tgt_idx

        # B x T x 1
        lambda_value = utils.softmax(torch.cat([s_knn, s_nmt], dim=2), dim=2)[:, :, :1]
        del s_knn, s_nmt

        temparature = torch.sigmoid(temparature)

        distance = distance.float()
        temparature = temparature.float()
        c_k = c_k.float()

        # note that the codes below should be `distance = - distance / temparature + c_k` as described in the paper, 
        # however, the official implementation uses the `*` operator instead of `/`, 
        # which can obtain better performance through our experiments. 
        # hence we use `- distance * temparature + c_k`
        distance = - distance * temparature + c_k

        distance = utils.softmax(distance, dim=2)
        distance = distance * lambda_value
        return {"knn_prob": distance, "tgt_idx": tgt_idx, "lambda_value": lambda_value}


class KernelSmoothedKnnSearch(KnnSearch):
    def __init__(self, cfg: KernelSmoothedKnnConfig):
        super().__init__(cfg)
    
        self.bandwidth_estimator = None
        self.weight_estimator = None

        assert self.cfg.load_keys, "please set `--load-keys` flag when using KernelSmoothedKnnSearch"
    
    def retrieve(self, queries):
        bsz, seq_len = queries.size()[:2]
        # B*T x C
        queries = queries.contiguous().view(-1, queries.size(-1))

        queries_device = queries.device

        if self.bandwidth_estimator.training:
            # B*T x K
            distance, idx = self.index.search(queries.cpu().float().numpy(), self.cfg.num_neighbors + 1)
            # retrieval dropout
            distance = distance[:, 1:]
            idx = idx[:, 1:]
        else:
            # B*T x K
            distance, idx = self.index.search(queries.cpu().float().numpy(), self.cfg.num_neighbors)
        
        # B*T x K x C
        retrieved_keys = torch.from_numpy(self.datastore_keys[idx]).type_as(queries)
        # B*T x 2*C
        features = torch.cat([queries, retrieved_keys.mean(dim=1)], dim=1)

        # B*T x 1
        bandwidth = self.bandwidth_estimator(features)
        bandwidth = bandwidth.exp()

        distance = torch.from_numpy(distance).type_as(queries)

        if self.cfg.kernel_type == "laplacian":
            distance.sqrt_()
        
        distance.neg_()
        
        distance = distance / bandwidth
        del bandwidth
        
        # B*T x K
        distance = utils.softmax(distance, dim=1)
        distance = distance.type_as(queries)

        # B*T x 1 x K @ B*T x K x C -> B*T x 1 x C
        weighted_retrieved_keys = distance.unsqueeze(1).matmul(retrieved_keys)
        del retrieved_keys

        # B*T x C
        weighted_retrieved_keys = weighted_retrieved_keys.squeeze(1)
        features = torch.cat([queries, weighted_retrieved_keys], dim=1)
        del queries, weighted_retrieved_keys

        # B*T x 1
        lambda_value = self.weight_estimator(features)

        # B x T x K
        distance = distance.view(bsz, seq_len, -1)

        # B x T x 1
        lambda_value = lambda_value.view(bsz, seq_len, -1)

        distance = distance * lambda_value
        tgt_idx = torch.from_numpy(self.datastore_values[idx]).to(queries_device)
        tgt_idx = tgt_idx.view(bsz, seq_len, -1)

        distance = distance.float()
        lambda_value = lambda_value.float()
        return {"knn_prob": distance, "tgt_idx": tgt_idx, "lambda_value": lambda_value}


def get_normalized_probs(
    task,
    model,
    net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
    log_probs: bool,
    sample: Optional[Dict[str, Tensor]] = None,
):
    """Get normalized probabilities (or log probs) from a net's output."""

    if hasattr(model.decoder, "adaptive_softmax") and model.decoder.adaptive_softmax is not None:
        if sample is not None:
            assert "target" in sample
            target = sample["target"]
        else:
            target = None
        mt_prob = model.decoder.adaptive_softmax.get_log_prob(net_output[0], target=target)
        mt_prob.exp_()
    else:
        mt_prob = net_output[0]
        mt_prob = utils.softmax(mt_prob, dim=-1)
    
    # T x B x C
    collected_keys = task.forward_hook.collected_outputs[0]
    task.forward_hook.clear()

    # B x T x C
    collected_keys = collected_keys.transpose(0, 1)
    search_results = task.knn_search.retrieve(collected_keys)

    lambda_value = search_results["lambda_value"]

    mt_prob.mul_(1.0 - lambda_value)
    mt_prob.scatter_add_(dim=2, index=search_results["tgt_idx"], src=search_results["knn_prob"])

    if log_probs:
        mt_prob.log_()

    return mt_prob


def get_captured_module(decoder, captured_module_name):
    PATTERN = re.compile(r"(.*?)\[(-\d+)\]")

    captured_module = decoder
    logger.info(f"Captured module name: {captured_module_name}")

    for attr in captured_module_name.split("."):
        match_obj = PATTERN.fullmatch(attr)

        if match_obj is None:
            module_name = attr
            idx = None
        else:
            module_name = match_obj.group(1)
            idx = int(match_obj.group(2))

        captured_module = getattr(captured_module, module_name)

        if idx is not None:
            captured_module = captured_module[idx]

    logger.info(f"Captured module: {captured_module}")

    return captured_module
