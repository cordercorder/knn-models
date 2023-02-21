import logging
import torch.nn as nn

from typing import (
    Dict, 
    List, 
    Tuple,
    Optional, 
)
from torch import Tensor
from functools import partial
from argparse import Namespace
from omegaconf import DictConfig
from dataclasses import dataclass
from fairseq import utils
from fairseq.tasks import register_task
from fairseq.tasks.translation import (
    TranslationTask,
    TranslationConfig,
)
from knn_models.dataclass import (
    RobustKnnConfig,
)
from knn_models.hook_utils import (
    ForwardHook,
)
from knn_models.knn_utils import (
    RobustKnnSearch,
    get_captured_module,
)


logger = logging.getLogger(__name__)


@dataclass
class TranslationRobustKnnConfig(TranslationConfig):
    """config for robust nearest neighbor machine translation"""
    knn_config: RobustKnnConfig = RobustKnnConfig()


@register_task("translation_robust_knn", dataclass=TranslationRobustKnnConfig)
class TranslationRobustKnnTask(TranslationTask):
    """task for robust nearest neighbor machine translation"""
    def __init__(self, cfg: TranslationRobustKnnConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.knn_search = RobustKnnSearch(cfg.knn_config)
        self.forward_hook = ForwardHook(cfg.knn_config.batch_first)
    
    def build_model(self, cfg, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)

        assert hasattr(model, "decoder"), \
            "TranslationRobustKnnTask only supports the model with decoder! " \
            f"There is no decoder in {model.__class__.__name__}."
        
        # freeze the parameters in the pretrained model
        for params in model.parameters():
            params.requires_grad = False
        
        module_names = ["W_1_5", "W_2", "W_3", "W_4", "W_6"]
        for _module, _module_name in zip(self.build_parameter_matrices(), module_names):
            model.add_module(_module_name, _module)

            # make these parameter matrices in the model and knn_search shared with each other
            assert hasattr(self.knn_search, _module_name)
            setattr(self.knn_search, _module_name, _module)
        
        # make the projection matrix in the model and knn_search shared with each other
        self.knn_search.output_projection = model.decoder.output_projection
        
        # rewrite `load_state_dict` function to successfully load the pretrained models 
        # when there are no parameters of these parameter matrices in the checkpoint
        model.load_state_dict = partial(load_state_dict, model)
        
        # collect outputs from the specified module in decoder as the datastore keys
        captured_module_name = self.cfg.knn_config.module_to_capture
        captured_module = get_captured_module(model.decoder, captured_module_name)
        captured_module.register_forward_hook(self.forward_hook.forward_hook_function)

        # rewrite `get_normalized_probs` function to support kNN augmented NMT
        model.get_normalized_probs = partial(get_normalized_probs, self, model)

        # rewrite `set_num_updates` function to get the number of training steps
        model.set_num_updates = partial(set_num_updates, model)
        return model

    def build_parameter_matrices(self):
        # we follow the description in the paper to name these parameter matrices
        
        # the top 8 probabilities of the NMT distribution are used 
        # as the input features, so the size of input feature will 
        # increase by 8
        W_6 = nn.Linear(self.cfg.knn_config.num_neighbors * 2 + 8, 1)

        W_4 = nn.Linear(2, self.cfg.knn_config.hidden_size_2)
        W_3 = nn.Linear(self.cfg.knn_config.hidden_size_2, 1)

        W_2 = nn.Linear(
            self.cfg.knn_config.num_neighbors * 2, 
            self.cfg.knn_config.hidden_size_1
        )
        # W_1_5 is composed of W_1 and W_5
        W_1_5 = nn.Linear(self.cfg.knn_config.hidden_size_1, 2)

        return W_1_5, W_2, W_3, W_4, W_6


def load_state_dict(
    model,
    state_dict,
    strict=True,
    model_cfg: Optional[DictConfig] = None,
    args: Optional[Namespace] = None,
):
    """`load_state_dict` function for TranslationRobustKnnTask"""
    model_state_dict = model.state_dict()
    for key in model_state_dict.keys():
        if "W_" in key and key not in state_dict:
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


def set_num_updates(model, num_updates):
    if not hasattr(model, "num_updates"):
        model.num_updates = 0
    else:
        model.num_updates = num_updates


def get_normalized_probs(
    task,
    model,
    net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
    log_probs: bool,
    sample: Optional[Dict[str, Tensor]] = None,
):
    """Get normalized probabilities (or log probs) from a net's output."""
    if sample is not None:
        assert "target" in sample
        target = sample["target"]
    else:
        target = None

    if hasattr(model.decoder, "adaptive_softmax") and model.decoder.adaptive_softmax is not None:
        raise Exception("Robust KNN-MT does not suppoert adaptive softmax yet")
    else:
        mt_prob = net_output[0]
        mt_prob = utils.softmax(mt_prob, dim=-1)
    
    # T x B x C
    collected_keys = task.forward_hook.collected_outputs[0]
    task.forward_hook.clear()

    # B x T x C
    collected_keys = collected_keys.transpose(0, 1)
    search_results = task.knn_search.retrieve(
        collected_keys, 
        mt_prob, 
        target, 
        model.num_updates,
        model.training,
    )

    lambda_value = search_results["lambda_value"]

    mt_prob.mul_(1.0 - lambda_value)
    mt_prob.scatter_add_(dim=2, index=search_results["tgt_idx"], src=search_results["knn_prob"])

    if log_probs:
        mt_prob.log_()

    return mt_prob
