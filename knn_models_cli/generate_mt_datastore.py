import ast
import logging
import os
import sys
from argparse import Namespace
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_generator import EnsembleModel


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


def get_4_gram_values(target):
    # target shape: B x T
    bsz = target.size(0)

    # B x T x 1
    target = target.unsqueeze(2)

    # [-1, x_1, x_2, x_3, ..., x_{n-1}]
    target_right_shift_1 = torch.cat(
        [target.new_full((bsz, 1, 1), -1), target[:, :-1]], 
        dim=1
    )

    # [-1, -1, x_1, x_2, x_3, ..., x_{n-2}]
    target_right_shift_2 = torch.cat(
        [target.new_full((bsz, 2, 1), -1), target[:, :-2]], 
        dim=1
    )

    # [-1, -1, -1, x_1, x_2, x_3, ..., x_{n-3}]
    target_right_shift_3 = torch.cat(
        [target.new_full((bsz, 3, 1), -1), target[:, :-3]], 
        dim=1
    )

    # B x T x 4
    # [[x_1, x_2, x_3, x_4, x_5, ..., x_n]
    #  [-1,  x_1, x_2, x_3, x_4, ..., x_{n-1}]
    #  [-1,  -1,  x_1, x_2, x_3, ..., x_{n-2}]
    #  [-1,  -1,  -1,  x_1, x_2, ..., x_{n-3}]]
    return torch.cat(
        [target, target_right_shift_1, target_right_shift_2, target_right_shift_3],
        dim=2
    )


def get_4_gram_values_probs(target_probs):
    # target_probs shape: B x T
    bsz = target_probs.size(0)

    # B x T x 1
    target_probs = target_probs.unsqueeze(2)

    # [inf, x_1, x_2, x_3, ..., x_{n-1}]
    target_probs_right_shift_1 = torch.cat(
        [target_probs.new_full((bsz, 1, 1), float("inf")), target_probs[:, :-1]], 
        dim=1
    )

    # [inf, inf, x_1, x_2, x_3, ..., x_{n-2}]
    target_probs_right_shift_2 = torch.cat(
        [target_probs.new_full((bsz, 2, 1), float("inf")), target_probs[:, :-2]], 
        dim=1
    )

    # [inf, inf, inf, x_1, x_2, x_3, ..., x_{n-3}]
    target_probs_right_shift_3 = torch.cat(
        [target_probs.new_full((bsz, 3, 1), float("inf")), target_probs[:, :-3]], 
        dim=1
    )

    # B x T x 4
    # [[x_1, x_2, x_3, x_4, x_5, ..., x_n]
    #  [inf, x_1, x_2, x_3, x_4, ..., x_{n-1}]
    #  [inf, inf,  x_1, x_2, x_3, ..., x_{n-2}]
    #  [inf, inf,  inf,  x_1, x_2, ..., x_{n-3}]]
    return torch.cat(
        [
            target_probs,
            target_probs_right_shift_1,
            target_probs_right_shift_2,
            target_probs_right_shift_3,
        ],
        dim=2
    )


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
            generate_datastore_keys = False
    
    if os.path.isfile(datastore_values_path):
        if over_write:
            logger.warning(f"{datastore_values_path} already exists! It will be overwritten!")
        else:
            # do not overwrite the old datastore values
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

    # additional 4 gram target and probability will be generated in the case of PCKMT datastore
    is_pckmt_datastore = os.environ.get("PCKMT_DATASTORE", None) is not None
    logger.info(f"Is PCKMT datastore: {is_pckmt_datastore}")

    if is_pckmt_datastore:
        generate_datastore_4_gram_values = True
        generate_datastore_4_gram_values_probs = True
    else:
        generate_datastore_4_gram_values = False
        generate_datastore_4_gram_values_probs = False

    datastore_4_gram_values_path = os.path.join(cfg.task.knn_config.datastore, "4_gram_values.npy")
    datastore_4_gram_values_probs_path = os.path.join(cfg.task.knn_config.datastore, "4_gram_values_probs.npy")

    if is_pckmt_datastore:
        if os.path.isfile(datastore_4_gram_values_path):
            if over_write:
                logger.warning(f"{datastore_4_gram_values_path} already exists! It will be overwritten!")
            else:
                # do not overwrite the old 4-gram datastore values
                generate_datastore_4_gram_values = False

        if os.path.isfile(datastore_4_gram_values_probs_path):
            if over_write:
                logger.warning(f"{datastore_4_gram_values_probs_path} already exists! It will be overwritten!")
            else:
                # do not overwrite the old 4-gram datastore value probabilities
                generate_datastore_4_gram_values_probs = False

    if generate_datastore_4_gram_values:
        logger.info(f"Saving 4-gram datastore values in {datastore_4_gram_values_path}")
        datastore_4_gram_values = np.memmap(
            datastore_4_gram_values_path,
            dtype=np.int64,
            mode="w+", 
            shape=(cfg.task.knn_config.datastore_size, 4)
        )
    else:
        datastore_4_gram_values = None
    
    if generate_datastore_4_gram_values_probs:
        logger.info(f"Saving 4-gram datastore value probabilities in {datastore_4_gram_values_probs_path}")
        datastore_4_gram_values_probs = np.memmap(
            datastore_4_gram_values_probs_path,
            dtype=np.float32,
            mode="w+", 
            shape=(cfg.task.knn_config.datastore_size, 4)
        )
    else:
        datastore_4_gram_values_probs = None

    if ((datastore_keys is None) and
        (datastore_values is None) and
        (datastore_4_gram_values is None) and
        (datastore_4_gram_values_probs is None)):
        logger.info("The datastore has already been saved! No datastore need to be generated")
        return
    
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

    pad_idx = task.target_dictionary.pad()

    # Initialize generator
    gen_timer = StopwatchMeter()

    # number of saved tokens in the datastore
    num_saved_tokens = 0

    # number of tokens in the dataset
    num_total_tokens = 0

    # whether to only only return features without applying output layer
    # only return features if there is no need to generate probabilities
    decoder_features_only = True if not generate_datastore_4_gram_values_probs else False

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
        
        target_probs = [] if not decoder_features_only else None

        with torch.no_grad():
            net_input = sample["net_input"]
            encoder_outs = ensemble_model.forward_encoder(sample["net_input"])
            
            encoder_out = None

            for i, model in enumerate(ensemble_model.models):
                if ensemble_model.has_encoder():
                    encoder_out = encoder_outs[i]
                
                decoder_out = model.decoder.forward(
                    net_input["prev_output_tokens"],
                    encoder_out=encoder_out,
                    features_only=decoder_features_only,
                )[0]

                if target_probs is not None:
                    # get probability
                    # B x T x V
                    decoder_out = F.softmax(decoder_out, dim=2)
                    target_probs.append(decoder_out)
                    del decoder_out

        if generate_datastore_keys:
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
            
            # B x T x C
            collected_keys = collected_keys.transpose(0, 1)
            # B*T x C
            collected_keys = collected_keys.contiguous().view(-1, collected_keys.size(2))
        else:
            task.forward_hook.clear()

        # B x T
        target = sample["target"]

        if generate_datastore_4_gram_values:
            # B x T x 4
            target_4_gram = get_4_gram_values(target)
        
        if generate_datastore_4_gram_values_probs:
            if len(target_probs) > 1:
                # B x T x V
                # If there is more than one model, taking the average probability as target probabilities
                target_probs = torch.stack(target_probs, dim=0).mean(dim=0)
            else:
                target_probs = target_probs[0]
            
            # B x T
            target_probs = target_probs.gather(dim=2, index=target.unsqueeze(2)).squeeze(2)

            # B x T x 4
            target_4_gram_probs = get_4_gram_values_probs(target_probs)
            del target_probs

        # B*T
        target = target.view(-1)
        # B*T
        target_mask = target.ne(pad_idx)
        
        # strip padding tokens
        non_pad_indices = torch.nonzero(target_mask, as_tuple=True)[0]

        if generate_datastore_keys:
            collected_keys = collected_keys.index_select(dim=0, index=non_pad_indices)
        
        target = target.index_select(dim=0, index=non_pad_indices)

        if generate_datastore_4_gram_values:
            # B*T x 4
            target_4_gram = target_4_gram.view(-1, 4)
            target_4_gram = target_4_gram.index_select(dim=0, index=non_pad_indices)
        
        if generate_datastore_4_gram_values_probs:
            # B*T X 4
            target_4_gram_probs = target_4_gram_probs.view(-1, 4)
            target_4_gram_probs = target_4_gram_probs.index_select(dim=0, index=non_pad_indices)

        if num_saved_tokens + ntokens > cfg.task.knn_config.datastore_size:
            effective_ntokens = cfg.task.knn_config.datastore_size - num_saved_tokens

            if generate_datastore_keys:
                collected_keys = collected_keys[: effective_ntokens]
            
            if generate_datastore_values:
                target = target[: effective_ntokens]
            
            if generate_datastore_4_gram_values:
                target_4_gram = target_4_gram[: effective_ntokens]
            
            if generate_datastore_4_gram_values_probs:
                target_4_gram_probs = target_4_gram_probs[: effective_ntokens]
        else:
            effective_ntokens = ntokens

        if generate_datastore_keys:
            collected_keys = collected_keys.cpu().numpy().astype(keys_dtype)
            datastore_keys[num_saved_tokens: num_saved_tokens + effective_ntokens] = collected_keys
        
        if generate_datastore_values:
            target = target.cpu().numpy().astype(np.int64)
            datastore_values[num_saved_tokens: num_saved_tokens + effective_ntokens] = target
        
        if generate_datastore_4_gram_values:
            target_4_gram = target_4_gram.cpu().numpy().astype(np.int64)
            datastore_4_gram_values[num_saved_tokens: num_saved_tokens + effective_ntokens] = target_4_gram
        
        if generate_datastore_4_gram_values_probs:
            target_4_gram_probs = target_4_gram_probs.cpu().numpy().astype(np.float32)
            datastore_4_gram_values_probs[num_saved_tokens: num_saved_tokens + effective_ntokens] = target_4_gram_probs
    
        num_saved_tokens += effective_ntokens

        gen_timer.stop(effective_ntokens)
        wps_meter.update(effective_ntokens)

        progress.log({"wps": round(wps_meter.avg)})

    # Flush the memmap instance to write the changes to the file
    if datastore_keys is not None:
        datastore_keys.flush()
    
    if datastore_values is not None:
        datastore_values.flush()
    
    if datastore_4_gram_values is not None:
        datastore_4_gram_values.flush()
    
    if datastore_4_gram_values_probs is not None:
        datastore_4_gram_values_probs.flush()

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
