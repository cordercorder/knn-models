import os
import sys
import logging
import numpy as np
import torch.nn.functional as F

try:
    from elasticsearch import (
        helpers,
        Elasticsearch,
    )
except ModuleNotFoundError:
    _elasticsearch_is_installed = False
else:
    _elasticsearch_is_installed = True


from typing import (
    Dict, 
    List, 
    Tuple,
    Iterable,
    Optional, 
)
from torch import Tensor
from fairseq import utils
from itertools import zip_longest


logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)
logging.getLogger('elastic_transport').setLevel(logging.WARNING)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)

logger = logging.getLogger("knn_models_cli.es_knn_utils")


def edit_distance(string1, string2):
    if len(string1) < len(string2):
        return edit_distance(string2, string1)

    # len(string1) >= len(string2)
    if len(string2) == 0:
        return len(string1)

    previous_row = range(len(string2) + 1)
    for i, c1 in enumerate(string1):
        current_row = [i + 1]
        for j, c2 in enumerate(string2):
            # j+1 instead of j since previous_row and current_row are one character longer than string2
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


class ElasticKnnSearch:
    def __init__(self, cfg):
        assert _elasticsearch_is_installed, \
            "Please install elasticsearch-py via `pip install elasticsearch`"

        self.client = Elasticsearch(
            cfg.hosts,
            ca_certs=cfg.ca_certs,
            basic_auth=("elastic", cfg.elastic_password),
        )

        if self.client.ping():
            logger.info("Successful connected to elasticsearch.")

    def create_index(self, index_name: str):
        settings = {
            "analysis": {
                "analyzer": {
                    "default": {
                        "type": "whitespace",
                    },
                    "default_search": {
                        "type": "whitespace",
                    }
                }
            },
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }

        mappings = {
            "properties":{
                "source_text": {
                    "type": "text",
                },
                "target_text": {
                    "type": "text",
                },
            }
        }

        response = self.client.indices.create(
            index=index_name,
            settings=settings,
            mappings=mappings
        )
        logger.info(f"Index create response: {response}")
    
    def get_index(self, index_name: str):
        for index in self.client.indices.get(index=index_name):
            logger.info(f"Index: {index}")
    
    def delete_index(self, index_name: str):
        if self.client.indices.exists(index=index_name):
            self.client.indices.delete(index=index_name)
            logger.info(f"Successfully delete index:{index_name}")
        else:
            logger.warning(f"Index {index_name} does not exist")
  
    def build_datasetore(self, index_name: str, source_corpus_path: str, target_corpus_path: str):

        def data_generator(source_corpus_path, target_corpus_path):
            fin_source = open(source_corpus_path, mode="r", encoding="utf-8")
            fin_target = open(target_corpus_path, mode="r", encoding="utf-8")

            for idx, (source_text, target_text) in enumerate(zip_longest(fin_source, fin_target)):
                assert source_text is not None
                assert target_text is not None

                source_text = source_text.strip()
                target_text = target_text.strip()

                yield {
                    "_op_type": "create",
                    "_index": index_name,
                    "_id": idx,
                    "_source": {
                        "source_text": source_text,
                        "target_text": target_text,
                    }
                }
            
            fin_source.close()
            fin_target.close()

        response = helpers.bulk(self.client, data_generator(source_corpus_path, target_corpus_path))
        logger.info(f"Bulk response: {response}")

        response = self.client.indices.refresh(index=index_name)
        logger.info(f"Refresh response: {response}")
        
        info = self.client.indices.stats(index=index_name)
        doc_count = info["indices"][index_name]["primaries"]["docs"]["count"]
        logger.info(f"Total document indexed {doc_count}")

    def retrieve(
        self, 
        queries: Iterable[str], 
        index_name: str, 
        size: int, 
        retrieve_source: bool, 
        re_rank: bool = False,
        num_sentences_retained: int = 1,
    ):
        searches = []
        for text in queries:
            searches.append(
                {
                    "index": index_name
                }
            )

            if retrieve_source:
                searches.append(
                    {
                        "query":{
                            "match": {
                                "source_text": text
                            }
                        },
                        "size": size
                    }
                )
            else:
                searches.append(
                    {
                        "query":{
                            "match": {
                                "target_text": text
                            }
                        },
                        "size": size
                    }
                )

        search_results = self.client.msearch(searches=searches)

        retrieved_source_text = []
        retrieved_target_text = []
        retrieved_text_ids = []
        for text, responses in zip(queries, search_results["responses"]):
            hits = responses["hits"]["hits"]

            retrieved_source_text_per_query = []
            retrieved_target_text_per_query = []
            retrieved_text_ids_per_query = []

            for neighbor in hits:
                _source = neighbor["_source"]
                retrieved_source_text_per_query.append(_source["source_text"])
                retrieved_target_text_per_query.append(_source["target_text"])
                retrieved_text_ids_per_query.append(neighbor["_id"])
            
            if re_rank:
                similarity_scores = []
                text_length = len(text)

                for retrieved_text in retrieved_source_text_per_query:
                    retrieved_text_length = len(retrieved_text)
                    _score = 1.0 - edit_distance(text, retrieved_text) / max(text_length, retrieved_text_length)

                    # sort in descending order
                    similarity_scores.append(-_score)
                
                similarity_scores = np.asarray(similarity_scores, dtype=np.float32)
                indices = np.argsort(similarity_scores)
                del similarity_scores

                indices = indices[: num_sentences_retained]

                retrieved_source_text_per_query_new = []
                retrieved_target_text_per_query_new = []
                retrieved_text_ids_per_query_new = []

                for idx in indices:
                    retrieved_source_text_per_query_new.append(retrieved_source_text_per_query[idx])
                    retrieved_target_text_per_query_new.append(retrieved_target_text_per_query[idx])
                    retrieved_text_ids_per_query_new.append(retrieved_text_ids_per_query[idx])
                
                del indices
                
                retrieved_source_text_per_query = retrieved_source_text_per_query_new
                del retrieved_source_text_per_query_new

                retrieved_target_text_per_query = retrieved_target_text_per_query_new
                del retrieved_target_text_per_query_new

                retrieved_text_ids_per_query = retrieved_text_ids_per_query_new
                del retrieved_text_ids_per_query_new

            retrieved_source_text.append(retrieved_source_text_per_query)
            retrieved_target_text.append(retrieved_target_text_per_query)
            retrieved_text_ids.append(retrieved_text_ids_per_query)

        return retrieved_source_text, retrieved_target_text, retrieved_text_ids


def convert_retrieved_text_to_tensor(retrieved_text, retrieved_text_ids, dictionary):
    tokens = []
    retrieved_text_ids_set= set()

    for retrieved_text_per_query, retrieved_text_ids_per_query in zip(
        retrieved_text, 
        retrieved_text_ids
    ):
        for line, retrieved_text_id in zip(
            retrieved_text_per_query, 
            retrieved_text_ids_per_query
        ):  
            # remove text with identical id in datastore
            if retrieved_text_id in retrieved_text_ids_set:
                continue
            
            retrieved_text_ids_set.add(retrieved_text_id)

            tokens.append(
                dictionary.encode_line(
                    line=line,
                    add_if_not_exist=False,
                ).long()
            )
    
    return tokens


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
    bsz, seq_len = collected_keys.size()[:2]

    # B*T x C
    collected_keys = collected_keys.contiguous().view(-1, collected_keys.size(2))

    # N x C
    datastore_keys = task.datastore_keys
    # N
    datastore_values = task.datastore_values

    datastore_keys_norm = task.datastore_keys_norm

    # ||X - Y||^2 = ||X||^2 + ||Y||^2 - 2 ||X|| * ||Y||
    # B*T x 1
    collected_keys_norm = collected_keys.pow(2).sum(1).unsqueeze(1)

    # B*T x N
    distance = collected_keys_norm + datastore_keys_norm - collected_keys.matmul(datastore_keys.T).mul(2)
    del collected_keys_norm, collected_keys

    # there may be some distance small than 0 (e.g., 1e-5) due to numeric errors, 
    # which results in inf in later computations
    distance.clamp_(min=0.0)

    # B*T x K
    distance, idx = distance.topk(task.cfg.es_knn_config.num_neighbors, dim=1, largest=False)

    # B x T x K
    distance = distance.view(bsz, seq_len, -1)

    temperature_value = task.cfg.es_knn_config.temperature_value
    
    # lambda = ReLU(1 - d0 / temperature)
    # B x T
    lambda_value = F.relu(1.0 - distance[:, :, 0].sqrt().div_(temperature_value))

    distance.neg_()
    distance.div_(temperature_value)
    distance = utils.softmax(distance, dim=-1)

    # B x T x 1
    lambda_value.unsqueeze_(2)

    distance.mul_(lambda_value)

    # B*T x K
    tgt_idx = datastore_values[idx]

    tgt_idx = tgt_idx.view(bsz, seq_len, -1)

    mt_prob.mul_(1.0 - lambda_value)
    del lambda_value

    mt_prob.scatter_add_(dim=2, index=tgt_idx, src=distance)

    if log_probs:
        mt_prob.log_()

    return mt_prob
