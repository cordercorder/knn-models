import ast
import logging
import os
import sys
from argparse import Namespace
from itertools import chain

import numpy as np
import torch
from omegaconf import DictConfig

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_generator import EnsembleModel

from knn_models.es_utils import (
    
)

logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
logger = logging.getLogger("knn_models_cli.generate_datastore")


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def main(cfg: DictConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for saving datastore!"

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    if cfg.generation.lm_path is not None:
        overrides["data"] = cfg.task.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [cfg.generation.lm_path], arg_overrides=overrides, task=None
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({cfg.task.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Create EnsembleModel
    ensemble_model = EnsembleModel(models)

    datastore_keys_path = os.path.join(cfg.task.knn_config.datastore, "keys.npy")
    datastore_values_path = os.path.join(cfg.task.knn_config.datastore, "values.npy")

    if os.path.isfile(datastore_keys_path):
        logger.warning(f"{datastore_keys_path} already exists! It will be overwritten!")
    
    if os.path.isfile(datastore_values_path):
        logger.warning(f"{datastore_values_path} already exists! It will be overwritten!")

    if cfg.task.knn_config.keys_dtype == "fp16":
        keys_dtype = np.float16
    else:
        keys_dtype = np.float32
    

    #! modified (不再需要使用numpy来存储feature)
    # try:
    #     keys_dimension = saved_cfg.model.decoder.embed_dim
    # except AttributeError:
    #     # legacy model config
    #     keys_dimension = saved_cfg.model.decoder_embed_dim
    
    # datastore_keys = np.memmap(
    #     datastore_keys_path, 
    #     dtype=keys_dtype, 
    #     mode="w+", 
    #     shape=(cfg.task.knn_config.datastore_size, keys_dimension)
    # )
    # datastore_values = np.memmap(
    #     datastore_values_path,
    #     dtype=np.int64,
    #     mode="w+",
    #     shape=(cfg.task.knn_config.datastore_size, )
    # )

    pad_idx = task.target_dictionary.pad()

    # Initialize generator
    gen_timer = StopwatchMeter()

    # number of saved tokens in the datastore
    num_saved_tokens = 0

    # number of tokens in the dataset
    num_total_tokens = 0

    wps_meter = TimeMeter()
    for sample in progress:
        ntokens = sample["ntokens"]
        num_total_tokens += ntokens

        if num_saved_tokens >= cfg.task.knn_config.datastore_size:
            continue

        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        gen_timer.start()
        
        with torch.no_grad():
            net_input = sample["net_input"]
            encoder_outs = ensemble_model.forward_encoder(sample["net_input"])
            
            encoder_out = None
            for i, model in enumerate(ensemble_model.models):
                if ensemble_model.has_encoder():
                    encoder_out = encoder_outs[i]
                
                model.decoder.forward(
                    net_input["prev_output_tokens"],
                    encoder_out=encoder_out,
                    features_only=True,
                )
        
        # [(T x B x C), ...]
        collected_keys = task.forward_hook.collected_outputs
        if len(collected_keys) > 1:
            # T x B x C
            # If there is more than one model, taking the average hidden states as datastore keys
            collected_keys = torch.stack(collected_keys, dim=0).mean(dim=0)
        else:
            # T x B x C
            collected_keys = collected_keys[0]
        
        task.forward_hook.clear()
        
        collected_keys = collected_keys.transpose(0, 1)
        target = sample["target"]

        target_mask = target.ne(pad_idx)
        non_zeros = torch.sum(target_mask, dim=1)

        collected_keys = collected_keys.cpu().numpy().astype(keys_dtype)
        target = target.cpu().numpy().astype(np.int64)        

        for index, (id, non_zero) in enumerate(zip(sample['id'], non_zeros)):
            id = id.item()
            non_zero = non_zero.item()

            es_connection.update_datastore()

        #! prev
        # # B x T x C
        # collected_keys = collected_keys.transpose(0, 1)
        # # B*T x C
        # collected_keys = collected_keys.contiguous().view(-1, collected_keys.size(2))

        # # B x T
        # target = sample["target"]
        # # B*T
        # target = target.view(-1)
        # # B*T
        # target_mask = target.ne(pad_idx)
        
        # # strip padding tokens
        # non_pad_indices = torch.nonzero(target_mask, as_tuple=True)[0]

        # collected_keys = collected_keys.index_select(dim=0, index=non_pad_indices)
        # target = target.index_select(dim=0, index=non_pad_indices)

        # if num_saved_tokens + ntokens > cfg.task.knn_config.datastore_size:
        #     effective_ntokens = cfg.task.knn_config.datastore_size - num_saved_tokens
        #     collected_keys = collected_keys[: effective_ntokens]
        #     target = target[: effective_ntokens]
        # else:
        #     effective_ntokens = ntokens
        
        # collected_keys = collected_keys.cpu().numpy().astype(keys_dtype)
        # target = target.cpu().numpy().astype(np.int64)

    
        # # datastore_keys[num_saved_tokens: num_saved_tokens + effective_ntokens] = collected_keys
        # # datastore_values[num_saved_tokens: num_saved_tokens + effective_ntokens] = target
        
        # num_saved_tokens += effective_ntokens

        gen_timer.stop(effective_ntokens)
        wps_meter.update(effective_ntokens)

        progress.log({"wps": round(wps_meter.avg)})

    # Flush the memmap instance to write the changes to the file
    # datastore_keys.flush()
    # datastore_values.flush()

    logger.info(
        "{:,} tokens in {:.1f}s, {:.2f} tokens/s)".format(
            gen_timer.n,
            gen_timer.sum,
            1.0 / gen_timer.avg,
        )
    )
    logger.info(
        "Saved tokens / All tokens: {} / {}".format(
            num_saved_tokens,
            num_total_tokens
        )
    )
    logger.info("Datastore keys dimension: {}".format(datastore_keys.shape[1]))


def cli_main():
    parser = options.get_generation_parser()
    # TODO: replace this workaround with refactoring of `AudioPretraining`
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="wav2vec2",
        help="Model architecture. For constructing tasks that rely on "
        "model args (e.g. `AudioPretraining`)",
    )
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
