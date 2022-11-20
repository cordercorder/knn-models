import sys
import json
import argparse

from argparse import Namespace
from knn_models.es_knn_utils import ElasticKnnSearch


def build_datasetore(args: Namespace):
    elastic_knn = ElasticKnnSearch(args)
    elastic_knn.create_index(args.index_name)
    elastic_knn.build_datasetore(
        args.index_name,
        args.source_corpus_path,
        args.target_corpus_path,
    )


def delete_index(args: Namespace):
    elastic_knn = ElasticKnnSearch(args)
    elastic_knn.delete_index(args.index_name)


def get_index(args: Namespace):
    elastic_knn = ElasticKnnSearch(args)
    elastic_knn.get_index(args.index_name)


def retrieve(args: Namespace):
    
    def query_generator():
        for text in args.queries:
            text = text.strip()
            yield text

    elastic_knn = ElasticKnnSearch(args)
    source_text_neighbors, target_text_neighbors = elastic_knn.retrieve(
        query_generator(),
        args.index_name,
        args.size,
        args.retrieve_source
    )

    retrieval_results = {}
    for query_idx, (source_text_neighbors_per_query, target_text_neighbors_per_query) in enumerate(
        zip(
            source_text_neighbors,
            target_text_neighbors,
        )
    ):
        combined_neighbors_per_query = {}
        for neighbor_idx, (source_text, target_text) in enumerate(
            zip(
                source_text_neighbors_per_query,
                target_text_neighbors_per_query
            )
        ):
            combined_neighbors_per_query[f"neighbor_{neighbor_idx}"] = (source_text, target_text)
        
        retrieval_results[f"query_{query_idx}"] = combined_neighbors_per_query
    
    del source_text_neighbors, target_text_neighbors
    
    with open(args.retrieval_results, mode="w", encoding="utf-8") as fout:
        json.dump(retrieval_results, fout)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hosts", 
        type=str, 
        default="https://localhost:9200", 
        help="https:ip/port"
    )

    # please refer https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html#elasticsearch-security-certificates
    # for more details about the CA certificate
    parser.add_argument(
        "--ca-certs", 
        type=str, 
        default=None, 
        help="path to the http_ca.crt security certificate"
    )
    parser.add_argument(
        "--elastic-password", 
        type=str, 
        default=None, 
        help="password for the elastic user"
    )
    parser.add_argument(
        "--operation", 
        type=str, 
        choices=["build_datasetore", "delete_index", "get_index", "retrieve"],
        required=True,
        help="operation to be accomplished"
    )

    parser.add_argument(
        "--index-name", 
        type=str, 
        required=True,
        help="name of index"
    )

    args, unknown = parser.parse_known_args()

    if args.operation == "build_datasetore":
        parser.add_argument(
            "--source-corpus-path", 
            type=str, 
            required=True,
            help="path to corpus of source languague"
        )
        parser.add_argument(
            "--target-corpus-path", 
            type=str, 
            required=True,
            help="path to corpus of target languague"
        )
    elif args.operation == "retrieve":
        parser.add_argument(
            "--queries",
            type=argparse.FileType(mode="r", encoding="utf-8"),
            default=sys.stdin,
            help="path to corpus to perform search. reading text from standard input by default"
        )
        parser.add_argument(
            "--size",
            type=int,
            default=1,
            help="number of hits to return"
        )
        parser.add_argument(
            "--retrieve-source",
            action="store_true",
            default=False,
            help="whether to retrieving according to source text"
        )
        parser.add_argument(
            "--retrieval-results",
            type=str,
            required=True,
            help="pato to retrieval results (json format)"
        )
    
    return parser


def cli_main():
    parser = get_parser()
    args = parser.parse_args()

    if args.operation == "build_datasetore":
        build_datasetore(args)
    elif args.operation == "delete_index":
        delete_index(args)
    elif args.operation == "get_index":
        get_index(args)
    elif args.operation == "retrieve":
        retrieve(args)


if __name__ == "__main__":
    cli_main()
