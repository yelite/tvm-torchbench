import threading
import contextlib

from torchdynamo.optimizations.backends import BACKENDS, create_backend
import tvm
from tvm.contrib.torch import optimize_torch
from tvm.meta_schedule import TuneConfig


def tvm_metaschedule(subgraph, tune_config: TuneConfig = None, work_dir: str = None):
    if subgraph.is_cuda:
        target = tvm.target.cuda()
    else:
        import multiprocessing
        target = tvm.target.Target(f"llvm --num-cores {multiprocessing.cpu_count()}")

    return optimize_torch(subgraph.scripted, subgraph.example_inputs,
                          target=target, tuning_config=tune_config, work_dir=work_dir)


if tvm_metaschedule.__name__ not in BACKENDS:
    create_backend(tvm_metaschedule)


def tvm_metaschedule_with_param(tune_config: TuneConfig = None, work_dir: str = None):
    def compiler(gm, example_inputs):
        return BACKENDS["tvm_metaschedule"](gm, example_inputs,
                                            tune_config=tune_config, work_dir=work_dir)
    return compiler
