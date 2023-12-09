from typing import Any, Dict
import timeit


class Benchmarkable:
    def __init__(self, name) -> None:
        self.name = name

    def __call__(self, data) -> None:
        pass


def benchmark(benchmark: Benchmarkable, data) -> Dict[str, Any]:
    benchmark(data)
    return {
        "name": benchmark.name,
        "train_time": 0,
        "test_time": 0,
        "train_accuracy": 0,
        "test_accuracy": 0,
        "max_memory_usage": 0,
    }


def train_fn():
    pass


def test_fn():
    pass
