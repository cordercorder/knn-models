import os
import sys
import logging

from typing import Iterable
from elasticsearch import (
    helpers,
    Elasticsearch,
)
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


class ElasticKnn:
    def __init__(self, cfg):
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

    def retrieve(self, index_name: str, queries: Iterable[str], num_neighbors: int, retrieve_source: bool):
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
                        "size": num_neighbors
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
                        "size": num_neighbors
                    }
                )

        search_results = self.client.msearch(searches=searches)

        source_text_neighbors = []
        target_text_neighbors = []
        for responses in search_results["responses"]:
            hits = responses["hits"]["hits"]

            source_text_neighbors_per_query = []
            target_text_neighbors_per_query = []

            for neighbor in hits:
                _source = neighbor["_source"]
                source_text_neighbors_per_query.append(_source["source_text"])
                target_text_neighbors_per_query.append(_source["target_text"])

            source_text_neighbors.append(source_text_neighbors_per_query)
            target_text_neighbors.append(target_text_neighbors_per_query)

        return source_text_neighbors, target_text_neighbors
