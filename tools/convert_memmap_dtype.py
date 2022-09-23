import os
import re
import sys
import math
import logging
import argparse
import numpy as np


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)


logger = logging.getLogger("convert_memmap_dtype")


def cast_memory_to_bytes(memory_string: str) -> float:
    """
    Parse a memory string and returns the number of bytes
    """
    conversion = {unit: (2 ** 10) ** i for i, unit in enumerate("BKMGTPEZ")}

    number_match = r"([0-9]*\.[0-9]+|[0-9]+)"
    unit_match = "("
    for unit in conversion:
        if unit != "B":
            unit_match += unit + "B|"
    for unit in conversion:
        unit_match += unit + "|"
    unit_match = unit_match[:-1] + ")"

    matching_groups = re.findall(number_match + unit_match, memory_string, re.IGNORECASE)

    if matching_groups and len(matching_groups) == 1 and "".join(matching_groups[0]) == memory_string:
        group = matching_groups[0]
        return float(group[0]) * conversion[group[1][0].upper()]

    raise ValueError(f"Unknown format for memory string: {memory_string}")


def _convert_memmap_dtype(
    keys_path,
    keys_size,
    keys_dimension,
    keys_dtype,
    output_keys_path,
    output_keys_dtype,
    memory_usage,
):
    if keys_dtype == output_keys_dtype:
        logger.info(
            "The dtype of origin datastore keys and output datastore keys is the same. "
            "There is no need to convert dtype."
        )
        return 
    
    KEYS_DTYPE_MAP = {"fp16": np.float16, "fp32": np.float32}

    memory_usage = cast_memory_to_bytes(memory_usage)
    if keys_dtype == "fp16":
        batch_size = math.floor(memory_usage / (keys_dimension * 2))
    else:
        batch_size = math.floor(memory_usage / (keys_dimension * 4))
    
    assert batch_size > 0

    logger.info(f"Set batch size to {batch_size}")

    keys = np.memmap(
        keys_path,
        dtype=KEYS_DTYPE_MAP[keys_dtype],
        mode="r",
        shape=(keys_size, keys_dimension)
    )

    output_keys_dtype_np = KEYS_DTYPE_MAP[output_keys_dtype]
    output_keys = np.memmap(
        output_keys_path,
        dtype=output_keys_dtype_np,
        mode="w+",
        shape=(keys_size, keys_dimension)
    )

    num_batches = keys_size // batch_size + int(keys_size % batch_size != 0)
    prev_idx = 0
    for _ in range(num_batches):
        cur_idx = min(prev_idx + batch_size, keys_size)
        output_keys[prev_idx: cur_idx] = keys[prev_idx: cur_idx].astype(output_keys_dtype_np)
        prev_idx = cur_idx
    
    logger.info("memmap dtype converting complete")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys-path", type=str, required=True)
    parser.add_argument("--keys-size", type=int ,required=True)
    parser.add_argument("--keys-dimension", type=int, required=True)
    parser.add_argument("--keys-dtype", type=str, default="fp16", choices=["fp16", "fp32"])

    parser.add_argument("--output-keys-path", type=str, required=True)
    parser.add_argument("--output-keys-dtype", type=str, choices=["fp16", "fp32"], required=True)

    parser.add_argument("--memory-usage", type=str, default="64G")
    return parser


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    _convert_memmap_dtype(
        args.keys_path,
        args.keys_size,
        args.keys_dimension,
        args.keys_dtype,
        args.output_keys_path,
        args.output_keys_dtype,
        args.memory_usage,
    )


if __name__ == "__main__":
    cli_main()
