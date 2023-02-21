import logging
import torch.nn as nn

from typing import Optional
from functools import partial
from argparse import Namespace
from omegaconf import DictConfig
from dataclasses import dataclass
from fairseq.tasks.translation import (
    TranslationTask,
    TranslationConfig,
)
from fairseq.tasks import register_task
from knn_models.dataclass import (
    KernelSmoothedKnnConfig,
)
from knn_models.hook_utils import (
    ForwardHook,
)
from knn_models.knn_utils import (
    KernelSmoothedKnnSearch,
    get_captured_module,
    get_normalized_probs,
)


logger = logging.getLogger(__name__)


@dataclass
class TranslationKernelSmoothedKnnConfig(TranslationConfig):
    """config for kernel smoothed nearest neighbor machine translation"""
    knn_config: KernelSmoothedKnnConfig = KernelSmoothedKnnConfig()


@register_task("translation_kernel_smoothed_knn", dataclass=TranslationKernelSmoothedKnnConfig)
class TranslationKernelSmoothedKnnTask(TranslationTask):
    """task for kernel smoothed nearest neighbor machine translation"""
    def __init__(self, cfg: TranslationKernelSmoothedKnnConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.knn_search = KernelSmoothedKnnSearch(cfg.knn_config)
        self.forward_hook = ForwardHook(cfg.knn_config.batch_first)

    def build_model(self, cfg, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)

        assert hasattr(model, "decoder"), \
            "TranslationKernelSmoothedKnnTask only supports the model with decoder! " \
            f"There is no decoder in {model.__class__.__name__}."
        
        # freeze the parameters in the pretrained model
        for params in model.parameters():
            params.requires_grad = False
        
        bandwidth_estimator, weight_estimator = self.build_estimator()
        model.add_module("bandwidth_estimator", bandwidth_estimator)
        model.add_module("weight_estimator", weight_estimator)

        # make the bandwidth_estimator and weight_estimator in the model and knn_search shared with each other
        self.knn_search.bandwidth_estimator = bandwidth_estimator
        self.knn_search.weight_estimator = weight_estimator

        # rewrite `load_state_dict` function to successfully load the pretrained models 
        # when there are no parameters of bandwidth_estimator and weight_estimator in the checkpoint
        model.load_state_dict = partial(load_state_dict, model)

        # collect outputs from the specified module in decoder as the datastore keys
        captured_module_name = self.cfg.knn_config.module_to_capture
        captured_module = get_captured_module(model.decoder, captured_module_name)
        captured_module.register_forward_hook(self.forward_hook.forward_hook_function)

        # rewrite `get_normalized_probs` function to support kNN augmented NMT
        model.get_normalized_probs = partial(get_normalized_probs, self, model)
        return model
        
    def build_estimator(self):
        keys_dimension = self.cfg.knn_config.keys_dimension
        bandwidth_estimator = nn.Linear(2 * keys_dimension, 1)
        weight_estimator = nn.Sequential(
            nn.Linear(2 * keys_dimension, keys_dimension),
            nn.ReLU(),
            nn.Linear(keys_dimension, 1),
            nn.Sigmoid()
        )
        return bandwidth_estimator, weight_estimator


def load_state_dict(
    model,
    state_dict,
    strict=True,
    model_cfg: Optional[DictConfig] = None,
    args: Optional[Namespace] = None,
):
    """`load_state_dict` function for TranslationKernelSmoothedKnnTask"""
    model_state_dict = model.state_dict()
    for key in model_state_dict.keys():
        if ("bandwidth_estimator" in key and key not in state_dict) or \
            ("weight_estimator" in key and key not in state_dict):
            logger.info(
                f"params {key} dose not exists in the pretrained model, "
                f"it will be initialized randomly"
            )
            state_dict[key] = model_state_dict[key]
    
    return super(model.__class__, model).load_state_dict(
        state_dict,
        strict,
        model_cfg,
        args,
    )
