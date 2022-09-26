import logging
import sys

import os
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm

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
logger = logging.getLogger("knn_models_cli.es_utils")


class Elastic(object):

    def __init__(self, args):

        '''
        self.args:
            1. self.source_lang: 
            2. self.target_lang:
            3. max_source_length:
            4. max_target_length:
            5. es_ip: 
            6. es_port:
            7. max_concurrent_searches:
        '''

        self.args = args

        self.client = Elasticsearch(
            f"http://{self.args.es_ip}:{self.args.es_port}",
            timeout=360
        ) 
        if self.client.ping():
            logger.info(" successful connected to localhost elasticsearch. ")
        
        self.source_lang = None
        self.target_lang = None

    def get_all_index(self):
        for index in self.client.indices.get_alias('*'):
            logger.info(f' index: {index}')

    def create_index(self, index_name):

        # TODO configuation should be optimization?
        body = {
            "settings":{
                "index":{
                    "analysis":{
                        "analyzer":"standard"
                    },
                    "number_of_shards": "1",
                    "number_of_replicas": "1",
                }
            },
            "mappings":{
                "properties":{
                    f"{self.source_lang}_data":{
                        "type": "text",
                        "similarity": "BM25",
                        "analyzer": "standard",
                    },
                    f"{self.target_lang}_data":{
                        "type": "text",
                        "similarity": "BM25",
                        "analyzer": "standard",
                    },
                }
            }
        }
        logger.info('start create the index.')
        self.client.indices.create(
            index=index_name,
            body=body,
            ignore=[400],
            )

        logger.info('successful create the index.')
    
    def delete_index(self, index_name):
        try:
            self.client.indices.delete(index=index_name)
            logger.info(f'successful delete index:{index_name}')
        except Exception:
            logger.info('elasticsearch index do not exist. ')
    
    def refresh_index(self, index_name):
        self.client.indices.refresh(index=index_name)

    def test_index(self, index_name):
        return self.client.indices.exists(index=index_name)

    def bulk_index(self, index_name, total_document):
        assert self.test_index(index_name)
        
        # TODO modified  'chunk_size' to args.
        success, _ = bulk(
            client=self.client, 
            actions=total_document,
            chunk_size=1000)
        logger.info(f'success insert number:{success}')

    def es_create_datasetore(self, index_name, source_data_path, target_data_path):

        if self.test_index(index_name):
            if self.args.overwriter_index:
                logger.info(f'index exist and will delete the {index_name} and overwrite it.')
                self.delete_index(index_name)
            else:
                logger.info('index already exist! if you want to re-init the index, please set --overwrite_index = True')
                return 
            
        self.create_index(index_name)

        total_source = open(source_data_path, 'r')
        total_target = open(target_data_path, 'r')

        total_document = []
        total_remove = 0

        for current_id, (source, target) in enumerate(zip(total_source, total_target)):

            # TODO should we also remove the lines that removed during fairseq-preprocess? (if not, es may get data not appear in the fairseq-data.) 
            # source_length = len(source.strip().split(' ')) 
            # target_length = len(target.strip().split(' '))
            # if source_length > self.args.max_source_length \
            #     or  target_length > self.args.max_target_length:
            #     logger.info(f'come to max position (source: {source_length}. target: {target_length}). drop it!')
            #     total_remove += 1
            #     continue
            
            document = {
                '_op_type': 'create',
                '_index': index_name,
                '_id': current_id,
                '_source':{
                    f'{self.source_lang}_data': source.strip(),
                    f'{self.target_lang}_data': target.strip(),
                }
            }
            total_document.append(document)

            # TODO should same to the 'chunk_size' args?
            if len(total_document) >= 1000:
                self.bulk_index(index_name=index_name, total_document=total_document)
                total_document = []
        
        total_source.close()
        total_target.close()
            
        if len(total_document) != 0:
            self.bulk_index(index_name=index_name, total_document=total_document)
        logger.info('push all the retrieval data into Elasticsearch!')

        self.client.indices.refresh(index=index_name)
        info = self.client.indices.stats(index=index_name)
        logger.info(f'total remove:{total_remove}')
        logger.info(f'total document indexed {info["indices"][index_name]["primaries"]["docs"]["count"]}') 

    def query_index(self, index_name, query_body):
        if isinstance(query_body, dict):
            # single query
            result = self.client.search(index=index_name, body=query_body)['hits']['hits']

            final_result = []
            final_ids = set()
            for item in result:
                temp_result = []
                temp_result.append(item['_source'][f'{self.source_lang}_data'])
                temp_result.append(item['_source'][f'{self.target_lang}_data'])
                final_ids.add(item['_id'])
                final_result.append(temp_result)

            return final_result, final_ids

        elif isinstance(query_body, list):
            # multi query
            result = self.client.msearch(
                index=index_name, 
                body=query_body,
                max_concurrent_searches=self.args.max_concurrent_searches)['responses']
            
            final_result = []
            final_ids = set()

            for item1 in result:
                item1 = item1['hits']['hits']
                temp1_result = []

                # NOTE we get the id through es. make sure id corresponding to the data in elasticsearch.
                for item2 in item1:
                    temp2_result = []
                    temp2_result.append(item2['_source'][f'{self.source_lang}_data'])
                    temp2_result.append(item2['_source'][f'{self.target_lang}_data'])
                    final_ids.add(int(item2['_id']))
                    temp1_result.append(temp2_result)

                final_result.append(temp1_result)

            return final_result, final_ids



