import logging
logger = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)
import os
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm


class Elastic(object):

    def __init__(self, args):

        '''
        self.args:
            1. source_lang: 
            2. target_lang:
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
            logging.info(" successful connected to localhost elasticsearch. ")

    def get_all_index(self):
        for index in self.client.indices.get_alias('*'):
            logging.info(f' index: {index}')

    def create_index(self, index_name):
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
                    f"{self.args.source_lang}_data":{
                        "type": "text",
                        "similarity": "BM25",
                        "analyzer": "standard",
                    },
                    f"{self.args.source_lang}_keys":{
                        "type": "text"
                    },
                    f"{self.args.source_lang}_values":{
                        "type": "text"
                    },
                    f"{self.args.target_lang}_data":{
                        "type": "text",
                        "similarity": "BM25",
                        "analyzer": "standard",
                    },
                    f"{self.args.target_lang}_keys":{
                        "type": "text"
                    },
                    f"{self.args.target_lang}_values":{
                        "type": "text"
                    }
                }
            }
        }
        logging.info('start create the index.')
        self.client.indices.create(
            index=index_name,
            body=body,
            ignore=[400],
            )

        logging.info('successful create the index.')
    
    def delete_index(self, index_name):
        try:
            self.client.indices.delete(index=index_name)
        except Exception:
            logging.info('elasticsearch index do not exist. ')
    
    def refresh_index(self, index_name):
        self.client.indices.refresh(index=index_name)

    def test_index(self, index_name):
        return self.client.indices.exists(index=index_name)

    def bulk_index(self, index_name, total_document):
        assert self.test_index(index_name)
        success, _ = bulk(
            client=self.client, 
            actions=total_document,
            chunk_size=1000)
        logging.info(f'success insert number:{success}')


    def train(self, index_name, source_data_path, target_data_path):

        if self.test_index(index_name=index_name):
            # self.delete_index(index_name=index_name)
            logger.info('index exist. ')
            return None

        self.create_index(index_name=index_name)

        total_source = open(source_data_path, 'r')
        total_target = open(target_data_path, 'r')

        total_document = []
        total_remove = 0

        for current_id, (source, target) in enumerate(tqdm(zip(total_source, total_target), desc='init es index', total=len(total_source))):

            source_length = len(source.strip().split(' ')) 
            target_length = len(target.strip().split(' '))

            if source_length > self.args.max_source_length \
                or  target_length > self.args.max_target_length:
                print(f'come to max position (source: {source_length}. target: {target_length}). drop it!')
                total_remove += 1
                continue
            
            document = {
                '_op_type': 'create',
                '_index': index_name,
                '_id': current_id,
                '_source':{
                    f'{self.args.source_lang}_data': source.strip(),
                    f'{self.args.target_lang}_data': target.strip(),
                }
            }
            total_document.append(document)
            if len(total_document) >= 1000:
                self.bulk_index(index_name=index_name, total_document=total_document)
                total_document = []
            
            
        if len(total_document) != 0:
            self.bulk_index(index_name=index_name, total_document=total_document)
        logging.info('push all the retrieval data into Elasticsearch!')

        self.client.indices.refresh(index=index_name)
        info = self.client.indices.stats(index=index_name)
        print(f'total remove:{total_remove}')
        print('total document indexed', info["indices"][index_name]["primaries"]["docs"]["count"]) 

    def query_index(self, index_name, query_body):
        if isinstance(query_body, dict):
            # single query
            result = self.client.search(index=index_name, body=query_body)['hits']['hits']

            final_result = []
            for item in result:
                temp_result = []
                temp_result.append(item['_source'][f'{self.args.source_lang}_data'])
                temp_result.append(item['_source'][f'{self.args.target_lang}_data'])
                temp_result.append(item['_source']['id'])
                final_result.append(temp_result)

            # [[source, target], [source, target], ...]
            return final_result

        elif isinstance(query_body, list):
            # multi query
            result = self.client.msearch(
                index=index_name, 
                body=query_body,
                max_concurrent_searches=self.args.max_concurrent_searches)['responses']
            
            final_result = []

            for item1 in result:
                item1 = item1['hits']['hits']
                temp1_result = []

                for item2 in item1:
                    temp2_result = []
                    temp2_result.append(item2['_score'])
                    temp2_result.append(item2['_source'][f'{self.args.source_lang}_data'])
                    temp2_result.append(item2['_source'][f'{self.args.target_lang}_data'])
                    temp1_result.append(temp2_result)

                final_result.append(temp1_result)

            return final_result


        

def es_create_datasetore(
    es_connection, 
    index_name, 
    source_data_path, 
    target_data_path, 
    args
    ):
  
    if args.overwrite_index:
        es_connection.delete_index(index_name)
    
    if es_connection.test_index(index_name):
        logger.info('index already exist!  if you want to re-init the index, \
                                        please set --overwrite_index = True')
        return

    es_connection.train(
        index_name,
        source_data_path,
        target_data_path,
    )


def es_update_datastore(
    es_connection, 
    index_name, 
    id, 
    data
    ):
    
    if not es_connection.test_inde(index_name):
        logger.info('index not found in Elasticsearch. make sure you create the index first!')
        return
    

