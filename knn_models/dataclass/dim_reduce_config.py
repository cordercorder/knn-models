from omegaconf import II
from typing import Optional
from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass, ChoiceEnum


DIM_REDUCE_METHOD_CHOICES = ChoiceEnum(["PCA"])


@dataclass
class DimReduceConfig(FairseqDataclass):
    dim_reduce_method: Optional[DIM_REDUCE_METHOD_CHOICES] = field(
        default=None,
        metadata={
            "help": "the method used for dimension reduction"
        }
    )
    pca_input_dim: int = field(
        default=0, 
        metadata={
            "help": "input dimension of PCA"
        }
    )
    pca_output_dim: int = II("task.knn_config.keys_dimension")
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
