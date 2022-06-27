import os
from typing import List
from dataclasses import dataclass

from tvm.meta_schedule import TuneConfig


def get_default_tune_config() -> TuneConfig:
    return TuneConfig(
        strategy="evolutionary",
        num_trials_per_iter=32,
        max_trials_per_task=0,
        max_trials_global=0,
    )


def get_model_tune_config(model_name: str) -> TuneConfig:
    return get_default_tune_config()


def get_benchmark_result_dir(model_name: str) -> str:
    return os.path.join("result", model_name)


@dataclass
class ModelBenchmarkConfig:
    model_name: str
    tune_config: TuneConfig
    result_dir: str


def get_all_model_config() -> List[ModelBenchmarkConfig]:
    result = []

    from torchbenchmark import _list_model_paths
    for model_path in _list_model_paths():
        model_name = os.path.basename(model_path)
        result.append(ModelBenchmarkConfig(
            model_name=model_name,
            tune_config=get_model_tune_config(model_name),
            result_dir=get_benchmark_result_dir(model_name)
        ))

    return result
