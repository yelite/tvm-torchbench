"""This file runs the torchdynamo/torchbench.py with TVM backend"""
import argparse
import importlib.util
import sys
import os
import logging
import functools

import torch
import torchdynamo

from dynamo_backend import tvm_metaschedule_with_param
from model_config import get_all_model_config


logger = logging.getLogger(__name__)


def load_dynamobench():
    spec = importlib.util.spec_from_file_location("dynamobench", "3rdparty/torchdynamo/torchbench.py")
    dynamobench = importlib.util.module_from_spec(spec)
    sys.modules["dynamobench"] = dynamobench
    os.chdir("3rdparty/torchdynamo")
    spec.loader.exec_module(dynamobench)
    os.chdir("../..")
    return dynamobench


dynamobench = load_dynamobench()


def setup_environment(target: str):
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)
    torch._C._jit_set_texpr_fuser_enabled(True)
    if torch.cuda.is_available():
        torch._C._jit_set_nvfuser_enabled(False)

    if target == "cuda":
        dynamobench.synchronize = torch.cuda.synchronize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", "-t", choices=["cpu", "cuda"], help="cpu or cuda")
    parser.add_argument(
        "--repeat", "-n", type=int, default=30, help="number of timing runs"
    )
    parser.add_argument(
        "--randomize-input",
        action="store_true",
        help="Whether to randomize the input values. Dimensions will be kept the same.",
    )

    args = parser.parse_args()
    target = args.target or "cpu"

    setup_environment(target)
    dynamobench.output_filename = os.path.join("result", "dynamo_speedup.csv")

    all_model_config = get_all_model_config()
    for model_config in all_model_config:
        tuning_work_dir = os.path.join(model_config.result_dir, "tuning")
        os.makedirs(tuning_work_dir, exist_ok=True)

        backend = tvm_metaschedule_with_param(
            tune_config=model_config.tune_config,
            work_dir=tuning_work_dir
        )
        experiment = functools.partial(dynamobench.speedup_experiment,
                                       args, dynamobench.forward_pass)
        device, name, model, example_inputs = dynamobench.load_model(target, model_config.model_name, is_training=False, use_eval_mode=True)
        dynamobench.run_one_model(
            name,
            model,
            is_training=False,
            model_iter_fn=dynamobench.forward_pass,
            example_inputs=example_inputs,
            optimize_ctx=torchdynamo.optimize(backend),
            experiment=experiment,
            cos_similarity=True,
            skip_accuracy_check=False
        )


if __name__ == "__main__":
    main()


