import os
import faiss
import torch
import logging
import numpy as np

from torch import Tensor
from fairseq import utils
from knn_models.dataclass import KnnConfig
from typing import Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


class KnnSearch:
    def __init__(self, cfg: KnnConfig):
        self.cfg = cfg
        self.setup_knn_search()

    def setup_knn_search(self):
        index_path = os.path.join(self.cfg.datastore, "faiss.index")
        index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)

        if self.cfg.device_id >= 0:
            
            logger.info(f"Putting index to GPU {self.cfg.device_id}")

            resource = faiss.StandardGpuResources()
            cloner_options = None
            if self.cfg.knn_fp16:
                cloner_options = faiss.GpuClonerOptions()
                cloner_options.useFloat16 = True
            
            index = faiss.index_cpu_to_gpu(provider=resource, device=self.cfg.device_id, index=index, options=cloner_options)
        
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
        
        if self.cfg.move_to_memory:
            logger.info("Moving datastore values into CPU memory")
            _datastore_values = np.empty((self.cfg.datastore_size, ), dtype=np.int64)
            _datastore_values[:] = datastore_values[:]
            datastore_values = _datastore_values

            if self.cfg.load_keys:
                logger.info("Moving datastore keys into CPU memory")
                _datastore_keys = np.empty((self.cfg.datastore_size, self.cfg.keys_dimension), dtype=keys_dtype)
                _datastore_keys[:] = datastore_keys[:]
                datastore_keys = _datastore_keys
        
        self.index = index
        self.datastore_values = datastore_values

        if self.cfg.load_keys:
            self.datastore_keys = datastore_keys

    def retrieve(self, queries):
        bsz, seq_len = queries.size()[:2]
        # B*T x C
        queries = queries.contiguous().view(-1, queries.size(-1))

        # B*T x K
        distance, idx = self.index.search(queries.detach().cpu().float().numpy(), self.cfg.num_neighbors)

        tgt_idx = torch.from_numpy(self.datastore_values[idx]).to(queries.device)
        tgt_idx = tgt_idx.view(bsz, seq_len, -1)

        # distance and queries should have the same device
        distance = torch.from_numpy(distance).to(queries.device)
        distance = distance.view(bsz, seq_len, -1)

        distance = distance.neg_().div_(self.cfg.temperature_value)
        distance = utils.softmax(distance, dim=-1)

        return {"knn_prob": distance, "tgt_idx": tgt_idx, "lambda_value": self.cfg.lambda_value}


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
        out = model.decoder.adaptive_softmax.get_log_prob(net_output[0], target=target)
        return out.exp_() if not log_probs else out

    mt_prob = net_output[0]

    # T x B x C
    collected_keys = task.forward_hook.collected_outputs[0]
    task.forward_hook.clear()

    # B x T x C
    collected_keys = collected_keys.transpose(0, 1)
    search_results = task.knn_search.retrieve(collected_keys)

    lambda_value = search_results["lambda_value"]

    mt_prob = utils.softmax(mt_prob, dim=-1)
    mt_prob.mul_(1.0 - lambda_value)
    mt_prob.scatter_add_(dim=2, index=search_results["tgt_idx"], src=search_results["knn_prob"].mul_(lambda_value))

    if log_probs:
        mt_prob.log_()

    return mt_prob
