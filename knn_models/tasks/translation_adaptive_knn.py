import math
import logging
import torch.nn as nn

from typing import Optional
from functools import partial
from argparse import Namespace
from dataclasses import dataclass
from omegaconf import DictConfig
from fairseq.tasks.translation import (
    TranslationTask,
    TranslationConfig,
)
from fairseq.tasks import register_task
from fairseq.dataclass import FairseqDataclass
from knn_models.dataclass import (
    AdaptiveKnnConfig,
    DimReduceConfig,
)
from knn_models.hook_utils import (
    ForwardHook,
    DimReduceForwardHook,
)
from knn_models.knn_utils import (
    AdaptiveKnnSearch,
    get_captured_module,
    get_normalized_probs,
)


logger = logging.getLogger(__name__)


@dataclass
class TranslationAdaptiveKnnConfig(TranslationConfig):
    """config for adaptive nearest neighbor machine translation"""
    knn_config: AdaptiveKnnConfig = AdaptiveKnnConfig()
    dim_reduce_config: DimReduceConfig = DimReduceConfig()


@register_task("translation_adaptive_knn", dataclass=TranslationAdaptiveKnnConfig)
class TranslationAdaptiveKnnTask(TranslationTask):
    """task for nearest neighbor machine translation"""
    def __init__(self, cfg: TranslationAdaptiveKnnConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.knn_search = AdaptiveKnnSearch(cfg.knn_config)

        if cfg.dim_reduce_config.dim_reduce_method is None:
            self.forward_hook = ForwardHook(cfg.knn_config.batch_first)
        else:
            self.forward_hook = DimReduceForwardHook(
                cfg.dim_reduce_config, cfg.knn_config.batch_first
            )

    def build_model(self, cfg: FairseqDataclass, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)

        assert hasattr(model, "decoder"), \
            "TranslationAdaptiveKnnTask only supports the model with decoder! " \
            f"There is no decoder in {model.__class__.__name__}."

        # freeze the parameters in the pretrained model
        for params in model.parameters():
            params.requires_grad = False

        # insert meta-k network into model
        meta_k_network = self.build_meta_k_network()
        model.add_module("meta_k_network", meta_k_network)

        # make the meta-k network in the model and knn_search shared with each other
        self.knn_search.meta_k_network = meta_k_network

        # rewrite `load_state_dict` function to successfully load the pretrained models 
        # when there are no parameters of meta-k network in the checkpoint
        model.load_state_dict = partial(load_state_dict, model)

        # collect outputs from the specified module in decoder as the datastore keys
        captured_module_name = self.cfg.knn_config.module_to_capture
        captured_module = get_captured_module(model.decoder, captured_module_name)
        captured_module.register_forward_hook(self.forward_hook.forward_hook_function)

        # rewrite `get_normalized_probs` function to support kNN augmented NMT
        model.get_normalized_probs = partial(get_normalized_probs, self, model)
        return model

    def build_meta_k_network(self):
        # get knn config
        cfg = self.cfg.knn_config

        # instantiate met-k network
        input_size = cfg.num_neighbors * 2
        # [0, 2^0, 2^1, ..., 2^k]
        output_size = int(math.log2(cfg.num_neighbors)) + 2

        meta_k_network = nn.Sequential(
            nn.Linear(input_size, cfg.meta_k_hidden_size),
            nn.Tanh(),
            nn.Dropout(p=cfg.meta_k_dropout),
            nn.Linear(cfg.meta_k_hidden_size, output_size)
        )

        # specific initialization
        nn.init.xavier_normal_(meta_k_network[0].weight[:, : cfg.num_neighbors], gain=0.01)
        nn.init.xavier_normal_(meta_k_network[0].weight[:, cfg.num_neighbors:], gain=0.1)

        return meta_k_network


def load_state_dict(
    model,
    state_dict,
    strict=True,
    model_cfg: Optional[DictConfig] = None,
    args: Optional[Namespace] = None,
):
    """`load_state_dict` function for TranslationAdaptiveKnnTask"""
    model_state_dict = model.state_dict()
    for key in model_state_dict.keys():
        if "meta_k_network" in key and key not in state_dict:
            logger.info(
                f"params {key} dose not exists in the pretrained model, "
                f"it will be initialized randomly"
            )
            state_dict[key] = model_state_dict[key]
    
    return super(model.__class__, model).load_state_dict(
        state_dict, 
        strict, 
        model_cfg, 
        args
    )
