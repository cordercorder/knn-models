import torch

from omegaconf import II
from typing import Optional
from functools import partial
from dataclasses import dataclass
from fairseq.tasks.translation import (
    TranslationTask,
    TranslationConfig,
)
from fairseq import utils
from fairseq.tasks import register_task
from fairseq.data import LanguagePairDataset
from fairseq.dataclass import FairseqDataclass
from knn_models.dataclass import (
    EsKnnConfig
)
from knn_models.hook_utils import (
    ForwardHook,
)
from knn_models.knn_utils import (
    get_captured_module,
)
from knn_models.es_knn_utils import (
    ElasticKnnSearch,
    get_normalized_probs,
    convert_retrieved_text_to_tensor,
)


@dataclass
class TranslationEsKnnConfig(TranslationConfig):
    """config for nearest neighbor machine translation"""
    es_knn_config: EsKnnConfig = EsKnnConfig()
    max_tokens: Optional[int] = II("dataset.max_tokens")
    batch_size: Optional[int] = II("dataset.batch_size")


@register_task("translation_es_knn", dataclass=TranslationEsKnnConfig)
class TranslationEsKnnTask(TranslationTask):
    """task for nearest neighbor machine translation"""
    def __init__(self, cfg: TranslationEsKnnConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.es_knn_search = ElasticKnnSearch(cfg.es_knn_config)
        self.forward_hook = ForwardHook()

        self.datastore_keys = None
        self.datastore_keys_norm = None
        self.datastore_values = None

    def build_model(self, cfg: FairseqDataclass, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)

        assert hasattr(model, "decoder"), \
            "TranslationEsKnnTask only supports the model with decoder! " \
            f"There is no decoder in {model.__class__.__name__}."
        
        # collect outputs from the specified module in decoder as the datastore keys
        captured_module_name = self.cfg.es_knn_config.module_to_capture
        captured_module = get_captured_module(model.decoder, captured_module_name)
        captured_module.register_forward_hook(self.forward_hook.forward_hook_function)

        # rewrite `get_normalized_probs` function to support kNN augmented NMT
        model.get_normalized_probs = partial(get_normalized_probs, self, model)
        return model

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        # B x T
        src_tokens = sample["net_input"]["src_tokens"]

        # check whether sample is on GPU for later use
        is_cuda = src_tokens.is_cuda

        src_tokens = src_tokens.cpu()
        queries = []

        pad_idx = self.source_dictionary.pad()
        for i in range(src_tokens.shape[0]):
            # strip padding first
            # then convert tensor to string
            queries.append(
                self.source_dictionary.string(
                    utils.strip_pad(
                        src_tokens[i, :], pad_idx
                    )
                )
            )
        del src_tokens

        # retrieve parallel sentence pairs according the the source sentences in sample
        retrieved_source_text, retrieved_target_text = self.es_knn_search.retrieve(
            queries,
            self.cfg.es_knn_config.index_name,
            self.cfg.es_knn_config.size,
            retrieve_source=True,
        )
        del queries

        # convert retrieved source texts to tensor
        retrieved_src_tokens = convert_retrieved_text_to_tensor(retrieved_source_text, self.source_dictionary)
        del retrieved_source_text

        retrieved_src_tokens_length = [t.numel() for t in retrieved_src_tokens]

        # convert retrieved target texts to tensor
        retrieved_tgt_tokens = convert_retrieved_text_to_tensor(retrieved_target_text, self.target_dictionary)
        del retrieved_target_text

        retrieved_tgt_tokens_length = [t.numel() for t in retrieved_tgt_tokens]

        retrieved_dataset = LanguagePairDataset(
            src=retrieved_src_tokens,
            src_sizes=retrieved_src_tokens_length,
            src_dict=self.source_dictionary,
            tgt=retrieved_tgt_tokens,
            tgt_sizes=retrieved_tgt_tokens_length,
            tgt_dict=self.target_dictionary,
            shuffle=False,
        )

        iterator = self.get_batch_iterator(
            dataset=retrieved_dataset,
            max_tokens=self.cfg.max_tokens,
            max_sentences=self.cfg.batch_size,
        ).next_epoch_itr(shuffle=False)

        # construct datastore using the retrieved parallel sentence pairs
        datastore_keys = []
        datastore_values = []

        # for simplicity, only use first model in models
        model = models[0]
        for batch in iterator:
            if is_cuda:
                batch = utils.move_to_cuda(batch)

            with torch.no_grad():
                model(
                    **batch["net_input"], 
                    return_all_hiddens=False, 
                    features_only=True
                )
            
            # T x B x C
            collected_keys = self.forward_hook.collected_outputs[0]
            self.forward_hook.clear()

            # B x T x C
            collected_keys = collected_keys.transpose(0, 1)

            # B*T x C
            collected_keys = collected_keys.contiguous().view(-1, collected_keys.size(2))

            # B*T
            target = batch["target"].view(-1)
            del batch

            # in most cases, the padding index between the source and target dictionary is the same
            # so we use the padding index of the source dictionary here
            target_mask = target.ne(pad_idx)

            # Reduced_B*T x C
            collected_keys = collected_keys[target_mask]
            # Reduced_B*T
            target = target[target_mask]
            del target_mask

            datastore_keys.append(collected_keys)
            datastore_values.append(target)

            del collected_keys, target

        # save the datastore_keys, datastore_keys_norm and datastore_values for later use
        datastore_keys = torch.cat(datastore_keys, dim=0)
        self.datastore_keys = datastore_keys
        self.datastore_keys_norm = datastore_keys.pow(2).sum(1)
        del datastore_keys

        self.datastore_values = torch.cat(datastore_values, dim=0)
        del datastore_values

        results = super().inference_step(generator, models, sample, prefix_tokens, constraints)

        # reset datastore_keys, datastore_keys_norm and datastore_values to None after decoding
        self.datastore_keys = None
        self.datastore_keys_norm = None
        self.datastore_values = None

        return results
