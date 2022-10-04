import argparse
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
import os
import shutil
import sys
import subprocess
import tempfile
import importlib.util


logger = logging.getLogger(__name__)

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ModelConfig:
    name: str
    skip: bool
    batch_size: int


# Special model list taken from https://github.com/pytorch/torchdynamo/blob/main/benchmarks/torchbench.py

# Some models have large dataset that doesn't fit in memory. Lower the batch
# size to test the accuracy.
USE_SMALL_BATCH_SIZE = {
    "demucs": 4,
    "densenet121": 4,
    "hf_Reformer": 4,
    "timm_efficientdet": 1,
}

SKIP = {
    # https://github.com/pytorch/torchdynamo/issues/101
    "detectron2_maskrcnn",
    # https://github.com/pytorch/torchdynamo/issues/145
    "fambench_xlmr",
}


def load_torchdynamo_benchmark_batchsize() -> Dict[str, int]:
    result = {}
    with open(
        os.path.join(
            CURRENT_DIRECTORY, "torchdynamo/benchmarks/torchbench_models_list.txt"
        )
    ) as f:
        for line in f:
            model_name, batch_size = line.split(",")
            result[model_name] = int(batch_size)
    return result


def get_all_models() -> List[ModelConfig]:
    models = []

    model_batch_size = load_torchdynamo_benchmark_batchsize()

    for model_name in os.listdir(
        os.path.join(CURRENT_DIRECTORY, "benchmark/torchbenchmark/models")
    ):
        batch_size = model_batch_size.get(model_name)
        if model_name in USE_SMALL_BATCH_SIZE:
            batch_size = USE_SMALL_BATCH_SIZE[model_name]

        models.append(
            ModelConfig(
                name=model_name,
                skip=model_name in SKIP,
                batch_size=batch_size,
            )
        )

    return models


def validate_passthrough_args(passthrough_args: List[str]):
    block_args = ["--work-dir", "--model", "--batch-size"]
    for block_arg in block_args:
        if block_arg in passthrough_args:
            raise RuntimeError(f"Passthrough args cannot have {block_arg}.")


def benchmark_model(
    model: ModelConfig,
    model_result_dir: str,
    passthrough_args: List[str],
    runner_script_path: str,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        args = [
            "python",
            runner_script_path,
            *passthrough_args,
            "--work-dir",
            tmpdir,
            "--model",
            model.name,
        ]
        if model.batch_size:
            args.extend(
                [
                    "--batch-size",
                    str(model.batch_size),
                ]
            )

        result = subprocess.run(
            args=args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        result.check_returncode()

        with open(os.path.join(tmpdir, "driver_stdout.log"), "w") as f:
            f.write(result.stdout)
        with open(os.path.join(tmpdir, "driver_stderr.log"), "w") as f:
            f.write(result.stderr)

        os.makedirs(model_result_dir)
        for f in os.listdir(tmpdir):
            shutil.move(os.path.join(tmpdir, f), model_result_dir)


def main():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--result-dir",
        type=str,
        default="result",
        help="The directory to store benchmark result.",
    )
    arg_parser.add_argument(
        "--runner-script",
        type=str,
        default=os.path.join(
            CURRENT_DIRECTORY,
            "tvm/python/tvm/meta_schedule/testing/torchbench/run.py",
        ),
        help="The path to the runner script.",
    )
    arg_parser.add_argument(
        "passthrough_args",
        nargs=argparse.REMAINDER,
        help="The args that are passed to the benchmark script.",
    )

    args = arg_parser.parse_args()
    passthrough_args = args.passthrough_args
    if passthrough_args and passthrough_args[0] == "--":
        del passthrough_args[0]
    validate_passthrough_args(passthrough_args)

    os.makedirs(args.result_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.result_dir, "run_all.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger.info("Start running model benchmarks.")
    logger.info("Logging into %s", os.path.abspath(args.result_dir))
    logger.info("Args to be passed to run.py: '%s'.", " ".join(passthrough_args))

    skipped = 0
    failed = 0
    finished = 0

    for model in get_all_models():
        if model.skip:
            skipped += 1
            logger.info("%s skipped.", model.name)
            continue

        model_result_dir = os.path.join(args.result_dir, model.name)
        if os.path.isdir(model_result_dir):
            skipped += 1
            logger.info(
                "%s skipped because it's already in the result directory.", model.name
            )
            continue

        logging.info("%s started.", model.name)
        try:
            benchmark_model(
                model, model_result_dir, passthrough_args, args.runner_script
            )
        except subprocess.CalledProcessError as e:
            failed += 1
            logger.error("%s failed with stderr: %s", model.name, e.stderr.strip())
            with open(os.path.join(args.result_dir, f"failed_{model.name}_stdout.log"), "w") as f:
                f.write(e.stdout)
            with open(os.path.join(args.result_dir, f"failed_{model.name}_stderr.log"), "w") as f:
                f.write(e.stderr)
        else:
            finished += 1
            logger.info("%s finished.", model.name)

    logger.info(
        "Finished all tasks. success: %d, failed: %d, skipped: %d",
        finished,
        failed,
        skipped,
    )


if __name__ == "__main__":
    main()
