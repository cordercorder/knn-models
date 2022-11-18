import json
import os
import re
import math
import faiss
import torch
import logging
import numpy as np
import time

from torch import Tensor
from fairseq import utils
from fairseq.data import data_utils
from fairseq.tasks.translation import TranslationTask

    
from knn_models.dataclass import (
    KnnConfig, 
    AdaptiveKnnConfig,
)

from knn_models.es_utils import (
    Elastic
)
from typing import Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


class SentenceSearch:

    def __init__(self, cfg: KnnConfig, connection: Elastic):

        self.cfg = cfg
        self.es_connection = connection
        # TODO should we set 'source_lang' & 'target_lang' to args?
        self.source_lang, self.target_lang = data_utils.infer_language_pair(self.cfg.sentence_retrieval_fairseq_data)

        self.es_connection.source_lang = self.source_lang
        self.es_connection.target_lang = self.target_lang

        # use during build sentence index. 
        self.batch_keys = None
        self.keys_dtype = np.float32 if self.cfg.keys_dtype == "fp32" else np.float16


        self.sentence_source_dict = TranslationTask.load_dictionary(
            f"{self.cfg.sentence_retrieval_fairseq_data}/dict.{self.source_lang}.txt"
        )
        self.sentence_target_dict = TranslationTask.load_dictionary(
            f"{self.cfg.sentence_retrieval_fairseq_data}/dict.{self.target_lang}.txt"
        )
        
        self.sentence_retrieval_data = data_utils.load_indexed_dataset(
            f"{self.cfg.sentence_retrieval_fairseq_data}/train.{self.source_lang}-{self.target_lang}.{self.target_lang}",
            self.sentence_target_dict,
            dataset_impl='mmap'
        )

         # TODO the bpe file name should fix?
        source_data_path =  f"{self.cfg.sentence_retrieval_bpe_data}/train.bpe.filtered.{self.source_lang}"
        target_data_path =  f"{self.cfg.sentence_retrieval_bpe_data}/train.bpe.filtered.{self.target_lang}"
        
        logger.info(f'source_bpe_data path to create index:{source_data_path}')
        logger.info(f'target_bpe_data path to create index:{target_data_path}')

        self.es_connection.es_create_datasetore(self.cfg.index_name, source_data_path, target_data_path)


        if not self.cfg.sentence_load_single:
            id2location_file = open(self.cfg.datastore + '/id2location.json', 'r')
            self.id2location = json.load(id2location_file)
            id2location_file.close()

            # NOTE similar to 'load_keys'. Here might need a lot memory.
            datastore_keys_path = os.path.join(self.cfg.datastore, "keys.npy")
            keys_dtype = np.float32 if self.cfg.keys_dtype == "fp32" else np.float16
            logger.info(f'Using sentence retrieval and load keys. Please make sure set the right datasetore_size ')
            logger.info(f"Loading datastore keys from {datastore_keys_path}")
            self.sentence_retrieval_keys = np.memmap(
                datastore_keys_path, 
                dtype=keys_dtype, 
                mode="r", 
                shape=(self.cfg.datastore_size, self.cfg.keys_dimension)
            )
            datastore_values_path = os.path.join(self.cfg.datastore, "values.npy")
            self.sentence_retrieval_values = np.memmap(
                datastore_values_path, 
                dtype=np.int64,
                mode="r",
                shape=(self.cfg.datastore_size, )
            )

    def build_sentence_index(self, sample):
        batch_body = []

        for src_tokens in sample['net_input']['src_tokens']:
            bpe_sentence = self.sentence_source_dict.string(src_tokens, extra_symbols_to_ignore=[self.sentence_source_dict.pad()]).strip()
            cur_body = [
                {'index': self.cfg.index_name},
                {
                    "query":{
                        "match":{
                            f"{self.source_lang}_data": bpe_sentence,
                        }
                    },
                    "size": self.cfg.sentence_retrieval_size,
                }
            ]
            batch_body.extend(cur_body)
        
        start_time = time.time()
        batch_result, _ = self.es_connection.query_index(self.cfg.index_name, batch_body)
        logger.info(f'query time:{time.time() - start_time}')

        start_time = time.time()
        batch_ids = set()
        # TODO. can use rouge_score to rerank?
        if self.cfg.use_sentence_rerank:
            try:
                from rapidfuzz import process
                from rapidfuzz.distance.Levenshtein import normalized_similarity
            except Exception:
                raise Exception('please install rapidfuzz for fuzz-rerank. (pip install rapidfuzz==2.0.11)')
            
            for cur_index, result in enumerate(batch_result):
                src_tokens = sample['net_input']['src_tokens'][cur_index]
                bpe_sentence = self.sentence_source_dict.string(src_tokens,extra_symbols_to_ignore=[self.sentence_source_dict.pad()]).strip()
                rerank_result = process.extract(
                        bpe_sentence,
                        [item[0] for item in result],
                        scorer=normalized_similarity,
                        limit=self.cfg.sentence_final_size,
                    )
                
                for rerank_item in rerank_result:
                    batch_ids.add(int(result[rerank_item[-1]][-1]))    
        else:
            try:
                assert self.cfg.sentence_retrieval_size == self.cfg.sentence_final_size
            except AssertionError:
                logger.info('if use-sentence-rerank == False. please set sentence-retrieval-size == sentence-final-size.')
            for cur_index, result in enumerate(batch_result):
                for index, item in enumerate(result):
                    if index >= self.cfg.sentence_final_size:
                        break
                    batch_ids.add(int(item[-1]))

        offset = 0
        batch_feature_lengthes = sum([len(self.sentence_retrieval_data[id]) for id in batch_ids])
        batch_keys = np.zeros([batch_feature_lengthes, self.cfg.keys_dimension], dtype=self.keys_dtype)
        batch_values = np.zeros([batch_feature_lengthes,], dtype=np.int64)

        if self.cfg.sentence_load_single:
            for cur_id in batch_ids:
                cur_key_path = self.cfg.datastore + f'/keys_{cur_id}.npy'
                cur_value_path = self.cfg.datastore + f'/values_{cur_id}.npy'
                cur_length = len(self.sentence_retrieval_data[cur_id])
                cur_keys = np.memmap(
                    cur_key_path, 
                    dtype=self.keys_dtype, 
                    mode="r", 
                    shape=(cur_length, self.cfg.keys_dimension)
                )
                # TODO. seems equal to self.sentence_retrieval_data[cur_id]
                cur_values = np.memmap(
                    cur_value_path,
                    dtype=np.int64,
                    mode="r",
                    shape=(cur_length, )
                )
                batch_keys[offset: offset + cur_length] = cur_keys
                batch_values[offset: offset + cur_length] = cur_values
                offset += cur_length
        else:
            for cur_id in batch_ids:
                start_location, end_location = self.id2location[str(cur_id)]
                cur_length = end_location - start_location
                batch_keys[offset: offset + cur_length] = self.sentence_retrieval_keys[start_location: end_location]
                batch_values[offset: offset + cur_length] = self.sentence_retrieval_values[start_location: end_location]
                offset += cur_length

        if self.keys_dtype == np.float16:
            batch_keys = batch_keys.astype(np.float32)
        self.batch_keys = torch.from_numpy(batch_keys).to(self.cfg.knn_device_id[0])
        batch_values = torch.from_numpy(batch_values).to(self.cfg.knn_device_id[0])

        # TODO should also build datastore_value_weights here?
        batch_values_weights = None
        logger.info(f'set key time:{time.time() - start_time}')
        return batch_values, batch_values_weights

    def search(self, queries, num_neighbors):
        norms_xq = (queries ** 2).sum(axis=1)
        norms_xb = (self.batch_keys ** 2).sum(axis=1)
        distances = norms_xq.reshape(-1, 1) + norms_xb -2 * queries @ self.batch_keys.T 
        return torch.topk(distances, num_neighbors, largest=False)

class KnnSearch:
    def __init__(self, cfg: KnnConfig):
        self.cfg = cfg

        if not cfg.saving_mode:
            if self.cfg.use_sentence_constraint:
                logger.info('use sentence constraint')
                es_connection = Elastic(args=self.cfg)
                self.index = SentenceSearch(self.cfg, es_connection) 
            else:
                self.setup_knn_search()

    def setup_sentence_search(self, sample):
        self.datastore_values, self.datastore_value_weights = self.index.build_sentence_index(sample)


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
        if self.cfg.use_sentence_constraint:
            distance, idx = self.index.search(queries, self.cfg.num_neighbors)
            tgt_idx = self.datastore_values[idx]
            tgt_idx = tgt_idx.view(bsz, seq_len, -1)
            distance = distance.view(bsz, seq_len, -1)
        else:
            distance, idx = self.index.search(queries.cpu().float().numpy(), self.cfg.num_neighbors)

            tgt_idx = torch.from_numpy(self.datastore_values[idx]).to(queries.device)
            tgt_idx = tgt_idx.view(bsz, seq_len, -1)

            # distance and queries should have the same device
            distance = torch.from_numpy(distance).to(queries.device)
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

        if self.cfg.use_sentence_constraint:
            distance, idx = self.index.search(queries, self.cfg.num_neighbors)
            tgt_idx = self.datastore_values[idx]
            tgt_idx = tgt_idx.view(bsz, seq_len, -1)
            distance = distance.view(bsz, seq_len, -1)
        else:
            # B*T x K
            distance, idx = self.index.search(queries.cpu().float().numpy(), self.cfg.num_neighbors)

            tgt_idx = torch.from_numpy(self.datastore_values[idx]).to(queries.device)
            tgt_idx = tgt_idx.view(bsz, seq_len, -1)

            # distance and queries should have the same device
            distance = torch.from_numpy(distance).to(queries.device)
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

        # B x T x R_k x k
        distance = distance.unsqueeze_(2).expand(bsz, seq_len, self.reduced_k, self.cfg.num_neighbors)
        self.distance_mask = self.distance_mask.to(distance.device)

        distance = distance.masked_fill(self.distance_mask, float("-inf"))

        distance = utils.softmax(distance, dim=-1)

        # B x T x k
        distance = p_meta.unsqueeze(2).matmul(distance).squeeze(2)

        return {"knn_prob": distance, "tgt_idx": tgt_idx, "lambda_value": lambda_value}

    def get_value_count(self, tgt_idx, relative_label_count=False):
        bsz, seq_len, k = tgt_idx.size()

        # B x T x k x k
        tgt_idx = tgt_idx.unsqueeze(2).expand(bsz, seq_len, k, k)

        self.value_count_mask = self.value_count_mask.to(tgt_idx.device)
        tgt_idx = tgt_idx.masked_fill(self.value_count_mask, -1)

        value_sorted = tgt_idx.sort(dim=3)[0]

        value_sorted[:, :, :, 1:].mul_(
            (value_sorted[:, :, :, 1:] - value_sorted[:, :, :, :-1]).ne_(0).long()
        )

        # B x T x k
        value_count = value_sorted.ne_(0).long().sum(dim=3)
        value_count[:, :, :-1].sub_(1)

        if relative_label_count:
            value_count[:, :, 1:] = value_count[:, :, 1:] - value_count[:, :, :-1]
        
        return value_count


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
