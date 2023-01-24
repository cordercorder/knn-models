from typing import Optional
from omegaconf import MISSING
from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass


@dataclass
class BaseEsKnnConfig(FairseqDataclass):
    hosts: str = field(
        default="https://localhost:9200",
        metadata={
            "help": "https:ip/port"
        }
    )

    # please refer https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html#elasticsearch-security-certificates
    # for more details about the CA certificate
    ca_certs: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to the http_ca.crt security certificate"
        }
    )
    elastic_password: Optional[str] = field(
        default=None,
        metadata={
            "help": "password for the elastic user"
        }
    )
    index_name: str = field(
        default=MISSING,
        metadata={
            "help": "name of index"
        }
    )
    size: int = field(
        default=10,
        metadata={
            "help": "the number of hits to return (the number of bilingual sentence pairs "
            "used for constructing the tiny datastore)"
        }
    )


@dataclass
class EsKnnConfig(BaseEsKnnConfig):
    module_to_capture: str = field(
        default="layers[-1]",
        metadata={
            "help": "the outputs of the module in decoder to be captured. "
            "the default module is the last layer of decoder"
        }
    )
    num_neighbors: int = field(
        default=1,
        metadata={
            "help": "the number of neighbors to retrieve"
        }
    )
    temperature_value: float = field(
        default=10,
        metadata={
            "help": "hyperparameter used for flattening the kNN probability distribution"
        }
    )
