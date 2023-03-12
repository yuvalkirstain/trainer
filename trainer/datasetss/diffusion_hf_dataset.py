from dataclasses import dataclass
from io import BytesIO

import torch
from PIL import Image
from torchvision import transforms
from datasets import load_from_disk, load_dataset, Dataset
from hydra.utils import instantiate
from omegaconf import II

from trainer.datasetss.base_dataset import BaseDataset, BaseDatasetConfig


def cat_batch(batch, column_name):
    return torch.cat([item[column_name] for item in batch], dim=0)


def stack_batch(batch, column_name):
    return torch.stack([item[column_name] for item in batch], dim=0)


@dataclass
class TokenizerConfig:
    _target_: str = "transformers.AutoTokenizer.from_pretrained"
    pretrained_model_name_or_path: str = II("model.pretrained_model_name_or_path")
    subfolder: str = "tokenizer"


@dataclass
class DiffusionHFDatasetConfig(BaseDatasetConfig):
    _target_: str = "trainer.datasetss.diffusion_hf_dataset.DiffusionHFDataset"
    dataset_name: str = "data/datasets/PickaPic_reward.ds"
    dataset_config_name: str = "null"
    image_column_name: str = "good_jpg"
    caption_column_name: str = "caption"
    from_disk: bool = True
    cache_dir: str = ".cache"

    tokenizer: TokenizerConfig = TokenizerConfig()

    batch_size: int = 2

    resolution: int = 512


def convert_to_rgb(example):
    return example.convert("RGB")


class DiffusionHFDataset(BaseDataset):

    def __init__(self, cfg: DiffusionHFDatasetConfig, split: str = "train"):
        self.cfg = cfg

        self.dataset = self.load_hf_dataset(split)

        self.tokenizer = instantiate(cfg.tokenizer)
        self.transform = transforms.Compose(
            [
                convert_to_rgb,
                transforms.Resize(cfg.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(cfg.resolution),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2. - 1.)
            ]
        )

    def load_hf_dataset(self, split: str) -> Dataset:
        if self.cfg.from_disk:
            dataset = load_from_disk(self.cfg.dataset_name, keep_in_memory=True)[split]
        else:
            dataset = load_dataset(
                self.cfg.dataset_name,
                self.cfg.dataset_config_name,
                cache_dir=self.cfg.cache_dir,
                split=split
            )
        return dataset

    def tokenize(self, example):
        caption = example[self.cfg.caption_column_name]
        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        return input_ids

    def process_image(self, image):
        image = Image.open(BytesIO(image))
        image = image.convert("RGB")
        pixel_values = self.transform(image)
        return pixel_values

    def __getitem__(self, idx):
        example = self.dataset[idx]
        input_ids = self.tokenize(example)

        pixel_values = self.process_image(example[self.cfg.image_column_name])

        item = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return item

    @staticmethod
    def collate_fn(batch):
        input_ids = cat_batch(batch, "input_ids")
        pixel_values = stack_batch(batch, "pixel_values")

        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        collated = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return collated

    def __len__(self):
        return len(self.dataset)
