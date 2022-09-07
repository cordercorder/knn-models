import os
import sys
import torch
import logging
import argparse
import tempfile
import subprocess

from subprocess import run
from typing import List, Tuple
from argparse import Namespace, REMAINDER


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)


logger = logging.getLogger("knn_models_cli.tune_knn_params")


def remove_duplicated_params(candidate_params: List[Tuple[int]]):
    deduped_params = []
    visited_params = set()
    
    for params in candidate_params:
        if params not in visited_params:
            deduped_params.append(params)
            visited_params.add(params)
        else:
            logger.warning(f"Duplicated hyper-parameters: {params}")
    
    return deduped_params


def get_sacrebleu_score(sys_output, reference, sacrebleu_args):
    # the BLEU score computed by fairseq is not accurate if there is UNK in data_bin
    # so we adopt the shell command of sacrebleu to compute BLEU score
    sacrebleu_args = sacrebleu_args.split()
    if not ("--score-only" in sacrebleu_args) and not ("-b" in sacrebleu_args):
        sacrebleu_args.append("--score-only")
    
    cmd = ["sacrebleu", reference, "-i", sys_output, *sacrebleu_args]
    output = run(cmd, capture_output=True).stdout
    output = output.decode()
    return float(output)


def main(args: Namespace):
    candidate_params = args.candidate_params if args.candidate_params is not None else []

    candidate_num_neighbors = args.candidate_num_neighbors if args.candidate_num_neighbors is not None else [None]
    candidate_lambda_value = args.candidate_lambda_value if args.candidate_lambda_value is not None else [None]
    candidate_temperature_value = args.candidate_temperature_value if args.candidate_temperature_value is not None else [None]

    sacrebleu_args = args.sacrebleu_args if args.sacrebleu_args is not None else ""

    for num_neighbors in candidate_num_neighbors:
        for lambda_value in candidate_lambda_value:
            for temperature_value in candidate_temperature_value:
                if num_neighbors is not None or lambda_value is not None or temperature_value is not None:
                    params = (num_neighbors, lambda_value, temperature_value)
                    candidate_params.append(params)

    candidate_params = remove_duplicated_params(candidate_params)
    num_tasks = len(candidate_params)
    logger.info(f"Number of tasks: {num_tasks}")
    logger.info(f"Candicate hyper-parameters: {candidate_params}")

    cuda_devices_count = torch.cuda.device_count()
    logger.info(f"{cuda_devices_count} CUDA devices detected")

    CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    if CUDA_VISIBLE_DEVICES is None:
        CUDA_VISIBLE_DEVICES = list(map(str, range(cuda_devices_count)))
    else:
        CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES.split(",")

    results = []
    current_task_id = 0

    while current_task_id < num_tasks:

        with tempfile.TemporaryDirectory() as tmpdir:
            
            processes = []

            for device_id in range(cuda_devices_count):

                if current_task_id >= num_tasks:
                    break

                current_env = os.environ.copy()
                current_env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES[device_id]

                cmd = [sys.executable, "-u", args.script]

                params = candidate_params[current_task_id]

                if params[0] is not None:
                    cmd.append(
                        f"--num-neighbors={params[0]}"
                    )
                
                if params[1] is not None:
                    cmd.append(
                        f"--lambda-value={params[1]}"
                    )
                
                if params[2] is not None:
                    cmd.append(
                        f"--temperature-value={params[2]}"
                    )
                
                f_out = open(os.path.join(tmpdir, f"raw_sys.task_{current_task_id}.out"), mode="w", encoding="utf-8")
                f_err = open(os.path.join(tmpdir, f"raw_sys.task_{current_task_id}.err"), mode="w", encoding="utf-8")
                
                logger.info(f"Task {current_task_id} start!")

                cmd.extend(args.remain_args)
                process = subprocess.Popen(cmd, env=current_env, stdout=f_out, stderr=f_err)
                processes.append(
                    (
                        current_task_id,
                        process,
                        f_out,
                        f_err,
                    )
                )
                current_task_id += 1

            for elem in processes:
                task_id, process, f_out, f_err = elem
                process.wait()
                
                f_out.close()
                f_err.close()

                logger.info(f"Task {task_id} complete!")

                if process.returncode != 0:
                    cmd = ["cat", f_out.name]
                    process_output = run(cmd, capture_output=True).stdout
                    process_output = process_output.decode()
                    logger.info(f"process output: {process_output}")

                    cmd = ["cat", f_err.name]
                    process_err = run(cmd, capture_output=True).stdout
                    process_err = process_err.decode()
                    logger.info(f"process error information: {process_err}")

                    raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
                
                raw_output = os.path.join(tmpdir, f"raw_sys.task_{task_id}.out")
                output = os.path.join(tmpdir, f"sys.task_{task_id}.out")

                cmd = f"cat {raw_output} | grep -P \"^D\" | sort -V | cut -f 3- > {output}"
                run(cmd, shell=True)
                bleu_score = get_sacrebleu_score(output, args.reference, sacrebleu_args)

                params = candidate_params[task_id]

                results.append(
                    {
                        "bleu_score": bleu_score,
                        "num_neighbors": params[0] if params[0] is not None else "default",
                        "lambda_value": params[1] if params[1] is not None else "default",
                        "temperature_value": params[2] if params[2] is not None else "default"
                    }
                )

    results.sort(key=lambda item: item["bleu_score"])
    for elem in results:
        logger.info(elem)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "script", 
        type=str, 
        help="the full path to the program which generates texts with a trained model"
    )
    parser.add_argument(
        "remain_args", 
        nargs=REMAINDER, 
        help="remain arguments"
    )

    parser.add_argument(
        "--reference", 
        type=str,
        required=True,
        help="reference file path"
    )

    # space separated candicate hyper-parameters.
    # the final tuned hyper-parameters are the cartesian product of 
    # `candidate_num_neighbors` ,`candidate_lambda_value` and `candidate_temperature_value`.
    parser.add_argument(
        "--candidate-num-neighbors", 
        type=int, 
        nargs="+", 
        default=None,
        help="the candicate values of `num_neighbors`"
    )
    parser.add_argument(
        "--candidate-lambda-value", 
        type=float, 
        nargs="+", 
        default=None,
        help="the candicate values of `lambda_value`"
    )
    parser.add_argument(
        "--candidate-temperature-value", 
        type=float, 
        nargs="+", 
        default=None,
        help="the candicate values of `temperature_value`"
    )

    # optionally, the complete hyper-parameters can be set by `candidate_params`.
    # `candidate_params` can be a list or tuple which is composed of triplets.
    # the first elements in triplet is a candicate value of `num_neighbors`.
    # the second elements in triplet is a candicate value of `lambda_value`.
    # the third elements in triplet is a candicate value of `temperature_value`.
    # the elements in triplet can be None, which means the corresponding hyper-parameter is ommited.
    parser.add_argument(
        "--candidate-params", 
        type=eval, 
        default=None, 
        help="the candidate hyper-parameters"
    )

    parser.add_argument(
        "--sacrebleu-args", 
        type=str, 
        help="arguments for sacrebleu"
    )
    
    return parser


def validate_args(args: Namespace):
    assert \
    args.candidate_num_neighbors is not None or \
        args.candidate_lambda_value is not None or \
            args.candidate_temperature_value is not None or \
                args.candidate_params is not None
    "At least one of `candidate_num_neighbors`, `candidate_lambda_value`, "
    "`candidate_temperature_value` and `candidate_params` must be set"

    if args.candidate_params is not None:
        assert isinstance(args.candidate_params, (list, tuple)), \
            "`candidate_params` must be an instance of list or tuple"
        
        for params in args.candidate_params:
            assert isinstance(params, (list, tuple)), \
                "The elements in `candidate_params` must be an instance of list or tuple"
            
            assert len(params) == 3, \
                "The length of elements in `candidate_params` must be 3."


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    validate_args(args)
    main(args)


if __name__ == "__main__":
    cli_main()
