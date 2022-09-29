import logging
import os
import sys
from argparse import Namespace

import torch
import numpy as np
from omegaconf import DictConfig

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter
from fairseq.sequence_scorer import SequenceScorer


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)


logger = logging.getLogger("knn_models_cli.generate_lm_datastore")


def main(cfg: DictConfig, **unused_kwargs):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    logger.info(cfg)

    if cfg.eval_lm.context_window > 0:
        # reduce tokens per sample by the required context window size
        cfg.task.tokens_per_sample -= cfg.eval_lm.context_window

    # Initialize the task using the current *cfg*
    task = tasks.setup_task(cfg.task)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [cfg.common_eval.path],
        arg_overrides=eval(cfg.common_eval.model_overrides),
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
        task=task,
    )

    # whether to overwrite the old datastore, default is False
    over_write = os.environ.get("OVER_WRITE", None) is not None
    logger.info(f"Whether to overwrite the old datastore: {over_write}")

    # whether to generate the datastore keys
    generate_datastore_keys = True

    # whether to generate the datastore values
    generate_datastore_values = True

    datastore_keys_path = os.path.join(cfg.task.knn_config.datastore, "keys.npy")
    datastore_values_path = os.path.join(cfg.task.knn_config.datastore, "values.npy")

    if os.path.isfile(datastore_keys_path):
        if over_write:
            logger.warning(f"{datastore_keys_path} already exists! It will be overwritten!")
        else:
            # do not overwrite the old datastore keys
            logger.warning(f"{datastore_keys_path} already exists! Skip generate datastore keys!")
            generate_datastore_keys = False
    
    if os.path.isfile(datastore_values_path):
        if over_write:
            logger.warning(f"{datastore_values_path} already exists! It will be overwritten!")
        else:
            # do not overwrite the old datastore values
            logger.warning(f"{datastore_values_path} already exists! Skip generate datastore values!")
            generate_datastore_values = False
    
    if generate_datastore_keys:
        if cfg.task.knn_config.keys_dtype == "fp16":
            keys_dtype = np.float16
        else:
            keys_dtype = np.float32
        
        # infer keys dimension from the saved config
        try:
            keys_dimension = saved_cfg.model.decoder.embed_dim
        except AttributeError:
            # legacy model config
            keys_dimension = saved_cfg.model.decoder_embed_dim
        
        logger.info(f"Saving datastore keys in {datastore_keys_path}")
        datastore_keys = np.memmap(
            datastore_keys_path, 
            dtype=keys_dtype, 
            mode="w+", 
            shape=(cfg.task.knn_config.datastore_size, keys_dimension)
        )
    else:
        datastore_keys = None

    if generate_datastore_values:
        logger.info(f"Saving datastore values in {datastore_values_path}")
        datastore_values = np.memmap(
            datastore_values_path,
            dtype=np.int64,
            mode="w+",
            shape=(cfg.task.knn_config.datastore_size, )
        )
    else:
        datastore_values = None
    
    if datastore_keys is None and datastore_values is None:
        logger.info("The datastore has already been saved! No datastore need to be generated")
        return

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu
    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    # Optimize ensemble for generation and set the source and dest dicts on the model
    # (required by scorer)
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    assert len(models) > 0

    logger.info(
        "num. model params: {:,}".format(sum(p.numel() for p in models[0].parameters()))
    )

    # Load dataset splits
    task.load_dataset(cfg.dataset.gen_subset)
    dataset = task.dataset(cfg.dataset.gen_subset)
    logger.info(
        "{} {} {:,} examples".format(
            cfg.task.data, cfg.dataset.gen_subset, len(dataset)
        )
    )

    itr = task.eval_lm_dataloader(
        dataset=dataset,
        max_tokens=cfg.dataset.max_tokens or 36000,
        batch_size=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            *[model.max_positions() for model in models]
        ),
        num_shards=max(
            cfg.dataset.num_shards,
            cfg.distributed_training.distributed_world_size,
        ),
        shard_id=max(
            cfg.dataset.shard_id,
            cfg.distributed_training.distributed_rank,
        ),
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
        context_window=cfg.eval_lm.context_window,
    )

    itr = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    source_dictionary = task.source_dictionary
    target_dictionary = task.target_dictionary

    if target_dictionary is None:
        target_dictionary = source_dictionary
    
    device = next(models[0].parameters()).device

    gen_timer = StopwatchMeter()
    scorer = SequenceScorer(target_dictionary, cfg.eval_lm.softmax_batch)

    # number of saved tokens in the datastore
    num_saved_tokens = 0

    for sample in itr:
        if "net_input" not in sample:
            continue

        sample = utils.move_to_cuda(sample, device=device)

        gen_timer.start()
        hypos = scorer.generate(models, sample)
        gen_timer.stop(sample["ntokens"])

        if generate_datastore_keys:
            # [(T x B x C), ...]
            # length: number of models
            collected_keys = task.forward_hook.collected_outputs
        else:
            collected_keys = None

        bsz = sample["target"].size(0)
        src_tokens = sample["net_input"]["src_tokens"]
        start_idxs = sample["start_indices"] if "start_indices" in sample else [0] * bsz

        for i, hypos_i in enumerate(hypos):
            hypo = hypos_i[0]

            tokens = hypo["tokens"]

            effective_tok_length = src_tokens.size(1) - start_idxs[i]
            if effective_tok_length == cfg.task.tokens_per_sample:
                # skip sample with incomplete context

                if generate_datastore_keys:
                    collected_keys_sample = [_keys[start_idxs[i]:, i] for _keys in collected_keys]
                    if len(collected_keys_sample) > 1:
                        # T x C
                        # If there is more than one model, taking the average hidden states as datastore keys
                        collected_keys_sample = torch.stack(collected_keys_sample, dim=0).mean(dim=0)
                    else:
                        collected_keys_sample = collected_keys_sample[0]
                    
                else:
                    collected_keys_sample = None

                if generate_datastore_values:
                    # T
                    target_sample = tokens
                else:
                    target_sample = None

                if num_saved_tokens + effective_tok_length > cfg.task.knn_config.datastore_size:
                    effective_tok_length = cfg.task.knn_config.datastore_size - num_saved_tokens

                    if generate_datastore_keys:
                        collected_keys_sample = collected_keys_sample[: effective_tok_length]

                    if generate_datastore_values:
                        target_sample = target_sample[: effective_tok_length]
                
                if generate_datastore_keys:
                    assert collected_keys_sample.size(0) == effective_tok_length
                    collected_keys_sample = collected_keys_sample.cpu().numpy().astype(keys_dtype)
                    datastore_keys[num_saved_tokens: num_saved_tokens + effective_tok_length] = collected_keys_sample
                
                if generate_datastore_values:
                    assert target_sample.size(0) == effective_tok_length
                    target_sample = target_sample.cpu().numpy().astype(np.int64)
                    datastore_values[num_saved_tokens: num_saved_tokens + effective_tok_length] = target_sample
                
                num_saved_tokens += effective_tok_length

            else:
                logger.info(f"Skip incomplete sample, length: {effective_tok_length}")
        
        task.forward_hook.clear()
    
    if datastore_keys is not None:
        datastore_keys.flush()
    
    if datastore_values is not None:
        datastore_values.flush()

    logger.info(
        "Evaluated {:,} tokens in {:.1f}s ({:.2f} tokens/s)".format(
            gen_timer.n, gen_timer.sum, 1.0 / gen_timer.avg if gen_timer.avg > 0 else 0
        )
    )
    logger.info("Saved tokens / All tokens: {} / {}".format(
        num_saved_tokens,
        gen_timer.n
    ))
    logger.info("Datastore keys dimension: {}".format(datastore_keys.shape[1]))


def cli_main():
    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(convert_namespace_to_omegaconf(args), main)


if __name__ == "__main__":
    cli_main()
