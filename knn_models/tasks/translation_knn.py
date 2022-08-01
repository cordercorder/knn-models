from functools import partial
from dataclasses import dataclass
from fairseq.tasks.translation import (
    TranslationTask,
    TranslationConfig,
)
from fairseq.tasks import register_task
from fairseq.dataclass import FairseqDataclass
from knn_models.dataclass import KnnConfig
from knn_models.hook_utils import ForwardHook
from knn_models.knn_utils import KnnSearch, get_normalized_probs


@dataclass
class TranslationKnnConfig(TranslationConfig):
    """config for nearest neighbor machine translation"""
    knn_config: KnnConfig = KnnConfig()


@register_task("translation_knn", dataclass=TranslationKnnConfig)
class TranslationKnnTask(TranslationTask):
    """task for nearest neighbor machine translation"""
    def __init__(self, cfg: TranslationKnnConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.forward_hook = ForwardHook()
        self.knn_search = KnnSearch(cfg.knn_config)

    def build_model(self, cfg: FairseqDataclass, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)

        assert hasattr(model, "decoder"), \
            "TranslationKnnTask only supports the model with decoder! " \
            f"There is no decoder in {model.__class__.__name__}."

        assert hasattr(model.decoder, "layers"), \
            "Since TranslationKnnTask collects outputs from the last layer " \
            "of decoder as the datastore keys by default, " \
            f"{model.__class__.__name__}.{model.decoder.__class__.__name__} should " \
            "has the `layers` attribute."
        
        # collect outputs from the last layer of decoder as the datastore keys
        model.decoder.layers[-1].register_forward_hook(self.forward_hook.forward_hook_function)

        # rewrite `get_normalized_probs` function to support kNN augmented NMT
        model.get_normalized_probs = partial(get_normalized_probs, self, model)
        return model
