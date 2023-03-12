from dataclasses import dataclass
from io import BytesIO
import torch
from PIL import Image
from accelerate.logging import get_logger
from datasets import load_from_disk, load_dataset, Dataset
from hydra.utils import instantiate
from omegaconf import II

from trainer.datasetss.base_dataset import BaseDataset, BaseDatasetConfig

logger = get_logger(__name__)


def simple_collate(batch, column_name):
    return torch.cat([item[column_name] for item in batch], dim=0)


@dataclass
class ProcessorConfig:
    _target_: str = "transformers.AutoProcessor.from_pretrained"
    pretrained_model_name_or_path: str = II("model.pretrained_model_name_or_path")


@dataclass
class CLIPHFDatasetConfig(BaseDatasetConfig):
    _target_: str = "trainer.datasetss.clip_hf_dataset.CLIPHFDataset"
    dataset_name: str = "data/datasets/PickaPic_reward.ds"
    dataset_config_name: str = "null"
    image_column_name: str = "good_jpg"
    bad_image_column_name: str = "bad_jpg"
    caption_column_name: str = "caption"
    from_disk: bool = True
    train_split_name: str = "train"
    valid_split_name: str = "validation_unique"
    test_split_name: str = "test_unique"
    cache_dir: str = ".cache"

    processor: ProcessorConfig = ProcessorConfig()


@dataclass
class DebugCLIPHFDatasetConfig(CLIPHFDatasetConfig):
    dataset_name: str = "yuvalkirstain/dog_small"
    from_disk: bool = False
    image_column_name: str = "image"
    bad_image_column_name: str = "image"


class CLIPHFDataset(BaseDataset):

    def __init__(self, cfg: CLIPHFDatasetConfig, split: str = "train"):
        self.cfg = cfg
        logger.info(f"Loading {split} dataset")
        self.dataset = self.load_hf_dataset(split)
        if self.cfg.limit_valid_size > 0:
            logger.info(f"Limiting valid size to {self.cfg.limit_valid_size}")
            self.dataset = self.dataset.select(range(self.cfg.limit_valid_size))
        logger.info(f"Loaded {len(self.dataset)} examples from {split} dataset")
        processor = instantiate(cfg.processor)
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor

    def load_hf_dataset(self, split: str) -> Dataset:
        if self.cfg.from_disk:
            dataset = load_from_disk(self.cfg.dataset_name)[split]
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
        pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return pixel_values

    def __getitem__(self, idx):
        example = self.dataset[idx]
        input_ids = self.tokenize(example)

        pixel_values = self.process_image(example[self.cfg.image_column_name])
        bad_pixel_values = self.process_image(example[self.cfg.bad_image_column_name])

        item = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "bad_pixel_values": bad_pixel_values,
        }
        return item

    @staticmethod
    def collate_fn(batch):
        input_ids = simple_collate(batch, "input_ids")
        pixel_values = simple_collate(batch, "pixel_values")
        bad_pixel_values = simple_collate(batch, "bad_pixel_values")

        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        bad_pixel_values = bad_pixel_values.to(memory_format=torch.contiguous_format).float()

        collated = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "bad_pixel_values": bad_pixel_values,
        }
        return collated

    def __len__(self):
        return len(self.dataset)
