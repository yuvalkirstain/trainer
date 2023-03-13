from dataclasses import dataclass

import torch


@dataclass
class BaseDatasetConfig:
    train_split_name: str = "train"
    valid_split_name: str = "validation"
    test_split_name: str = "test"

    batch_size: int = 4
    num_workers: int = 8
    drop_last: bool = True

    limit_valid_size: int = -1


class BaseDataset(torch.utils.data.Dataset):
    pass
