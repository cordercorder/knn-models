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
    nprobe: int = field(
        default=32,
        metadata={
            "help": "the number of clusters to query"
        }
    )
    device_id: int = field(
        default=0,
        metadata={
            "help": "ID of GPU device used for (approximate) knn search. "
            "a negtive number means using CPU instead of GPU. "
            "note that this device can be different from the one used for translation."
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
