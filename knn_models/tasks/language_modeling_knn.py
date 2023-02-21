from functools import partial
from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks.language_modeling import (
    LanguageModelingTask, 
    LanguageModelingConfig,
)
from fairseq.tasks import register_task
from knn_models.dataclass import (
    KnnConfig, 
    DimReduceConfig,
)
from knn_models.hook_utils import (
    ForwardHook, 
    DimReduceForwardHook,
)
from knn_models.knn_utils import (
    KnnSearch, 
    get_captured_module,
    get_normalized_probs,
)


@dataclass
class LanguageModelingKnnConfig(LanguageModelingConfig):
    """config for nearest neighbor language modeling"""
    knn_config: KnnConfig = KnnConfig()
    dim_reduce_config: DimReduceConfig = DimReduceConfig()
    score_knn_lm: bool = field(
        default=False,
        metadata={"help": "wether to score the k-nearest neighbor language model with `eval_knn_lm`"}
    )
    recompute_distance: bool = field(
        default=False,
        metadata={
            "help": "wether to recompute distance for more accurate knn probability. As recomputing "
            "distance needs to load the datastore keys, `--load-keys` must be set."
        }
    )


@register_task("language_modeling_knn", dataclass=LanguageModelingKnnConfig)
class LanguageModelingKnnTask(LanguageModelingTask):
    def __init__(self, args, dictionary, output_dictionary=None, targets=None):
        super().__init__(args, dictionary, output_dictionary, targets)
        self.knn_search = KnnSearch(args.knn_config)

        if args.recompute_distance:
            assert args.knn_config.load_keys, \
                "Recomputing distance needs to load the datastore keys. `--load-keys` must be set."

        if args.dim_reduce_config.dim_reduce_method is None:
            self.forward_hook = ForwardHook(args.knn_config.batch_first)
        else:
            self.forward_hook = DimReduceForwardHook(
                args.dim_reduce_config, args.knn_config.batch_first
            )

    def build_model(self, cfg: FairseqDataclass, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)

        assert hasattr(model, "decoder"), \
            "LanguageModelingKnnTask only supports the model with decoder! " \
            f"There is no decoder in {model.__class__.__name__}."

        # collect outputs from the specified module of decoder as the datastore keys
        captured_module_name = self.args.knn_config.module_to_capture
        captured_module = get_captured_module(model.decoder, captured_module_name)
        captured_module.register_forward_hook(self.forward_hook.forward_hook_function)

        if not self.args.knn_config.saving_mode and not self.args.score_knn_lm:
            # rewrite `get_normalized_probs` function to support kNN augmented LM
            model.get_normalized_probs = partial(get_normalized_probs, self, model)
        
        return model
