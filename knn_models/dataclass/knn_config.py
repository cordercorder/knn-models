from curses import meta
from email.policy import default
from typing import List
from importlib_metadata import metadata
from omegaconf import MISSING
from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass, ChoiceEnum


KEYS_DTYPE_CHOICES = ChoiceEnum(["fp16", "fp32"])


@dataclass
class BaseKnnConfig(FairseqDataclass):
    datastore: str = field(
        default=MISSING,
        metadata={
            "help": "path to datastore directory"
        }
    )
    datastore_size: int = field(
        default=0,
        metadata={
            "help": "the number of tokens in datastore"
        }
    )
    keys_dimension: int = field(
        default=0,
        metadata={
            "help": "the feature dimension of datastore keys"
        }
    )
    keys_dtype: KEYS_DTYPE_CHOICES = field(
        default="fp16",
        metadata={
            "help": "keys dtype of the datastore"
        }
    )
    load_keys: bool = field(
        default=False,
        metadata={
            "help": "whether to load datastore keys"
        }
    )
    load_value_weights: bool = field(
        default=False,
        metadata={
            "help": "whether to load the weights of datastore values"
        }
    )
    nprobe: int = field(
        default=32,
        metadata={
            "help": "the number of clusters to query"
        }
    )
    knn_device_id: List[int] = field(
        default_factory=lambda: [0],
        metadata={
            "help": "ID of GPU device used for (approximate) knn search. "
            "a single negtive number means using CPU instead of GPU. "
            "note that this device can be different from the one used for translation."
            "if there is more than one number in `knn_device_id`, all numbers must "
            "be greater or equal to zero and the faiss index will be sharded across "
            "the GPU devices specified by `knn_device_id`. "
            "eg., --knn-device-id '1,2,3' "
        }
    )
    knn_fp16: bool = field(
        default=False,
        metadata={
            "help": "whether to perform intermediate calculations in float16 during (approximate) knn search"
        }
    )
    move_to_memory: bool = field(
        default=False,
        metadata={
            "help": "whether to move the datastore into CPU memory"
        }
    )
    module_to_capture: str = field(
        default="layers[-1]",
        metadata={
            "help": "the outputs of the which module in decoder to be captured. "
            "the default module is the last layer of decoder"
        }
    )
    saving_mode: bool = field(
        default=False,
        metadata={
            "help": "whether to use saving mode. "
            "the knn search setup process will be skipped in saving mode. "
            "saving mode is usually used when saving datastore"
        }
    )

    use_sentence_constraint: bool = field(
        default=False,
        metadata={
            "help": "Whether to use sentence-level constraint!"
            "original KNN use index at token-level. By set use_sentence_constraint=True,"
            "we can use the sentence-level search for every sentences, and formulate a new index during every batch."
        }
    )

    index_name: str = field(
        default='koran',
        metadata={
            "help": "The index in elasticsearch to save the raw retrieval data."
        }
    )

    overwriter_index: bool = field(
        default=False,
        metadata={
            "help": "whether to overwrite the create index."
        }
    )

    es_ip: str = field(
        default='localhost',
        metadata={
            "help": "the ip to connected to Elasticsearch"
        }
    )

    es_port: str = field(
        default='9200',
        metadata={
            "help": "the port to connected to Elasticsearch",
        }
    )

    max_concurrent_searches: int = field(
        default=64,
        metadata={
            "help": "max concurrent search during es search. might related to you CPUs."
        }
    )


@dataclass
class KnnConfig(BaseKnnConfig):
    num_neighbors: int = field(
        default=1,
        metadata={
            "help": "the number of neighbors to retrieve"
        }
    )
    lambda_value: float = field(
        default=0.5,
        metadata={
            "help": "hyperparameter used for interpolation of kNN and MT probability distributions"
        }
    )
    temperature_value: float = field(
        default=10,
        metadata={
            "help": "hyperparameter used for flattening the kNN probability distribution"
        }
    )


@dataclass
class AdaptiveKnnConfig(BaseKnnConfig):
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
    meta_k_hidden_size: int = field(
        default=32,
        metadata={
            "help": "hidden size of meta-k network"
        }
    )
    meta_k_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout of meta-k network"
        }
    )
