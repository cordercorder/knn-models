from omegaconf import II
from typing import Optional
from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass, ChoiceEnum


DIM_REDUCE_METHOD_CHOICES = ChoiceEnum(["PCA", "PCKMT"])


@dataclass
class DimReduceConfig(FairseqDataclass):
    dim_reduce_method: Optional[DIM_REDUCE_METHOD_CHOICES] = field(
        default=None,
        metadata={
            "help": "the method used for dimension reduction"
        }
    )
    datastore: str = II("task.knn_config.datastore")
    transform_ckpt_name: str = field(
        default="transform.pt",
        metadata={
            "help": "checkpoint filename of the saved module which is "
            "used for transformation (dimension reduction). "
            "the path to the checkpoint is `datastore`/`transform_ckpt_name`"
        }
    )
    cpu: bool = II("common.cpu")

    # configuration for dimension reduction with PCA
    pca_input_size: int = field(
        default=0, 
        metadata={
            "help": "input feature dimension of PCA"
        }
    )
    pca_output_size: int = II("task.knn_config.keys_dimension")

    # configuration for dimension reduction with compact network
    compact_net_input_size: int = field(
        default=0,
        metadata={
            "help": "input feature dimension of compact network"
        }
    )
    compact_net_hidden_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "hidden size of compact network. "
            "it will be set to `keys_dimension`/4 by default"
        }
    )
    compact_net_output_size: int = II("task.knn_config.keys_dimension")
