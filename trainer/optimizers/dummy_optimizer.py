from dataclasses import dataclass


@dataclass
class DummyOptimizerConfig:
    _target_: str = "accelerate.utils.DummyOptim"
    lr: float = 1e-6
