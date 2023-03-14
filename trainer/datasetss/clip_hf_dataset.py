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
    dataset_name: str = "data_all/datasets/PickaPic_reward.ds"
    dataset_config_name: str = "null"

    from_disk: bool = True
    train_split_name: str = "train"
    valid_split_name: str = "validation_unique"
    test_split_name: str = "test_unique"
    cache_dir: str = ".cache"

    caption_column_name: str = "caption"
    input_ids_column_name: str = "input_ids"
    image_0_column_name: str = "jpg_0"
    image_1_column_name: str = "jpg_1"
    label_0_column_name: str = "label_0"
    label_1_column_name: str = "label_1"
    are_different_column_name: str = "are_different"
    has_label_column_name: str = "has_label"

    pixels_0_column_name: str = "pixel_values_0"
    pixels_1_column_name: str = "pixel_values_1"

    num_examples_per_prompt_column_name: str = "num_example_per_prompt"

    keep_only_different: bool = False
    keep_only_with_label: bool = False

    processor: ProcessorConfig = ProcessorConfig()


class CLIPHFDataset(BaseDataset):

    def __init__(self, cfg: CLIPHFDatasetConfig, split: str = "train"):
        self.cfg = cfg
        logger.info(f"Loading {split} dataset")

        self.dataset = self.load_hf_dataset(split)

        if self.cfg.keep_only_different:
            self.dataset = self.dataset.filter(lambda x: x[self.cfg.are_different_column_name])
        if self.cfg.keep_only_with_label:
            self.dataset = self.dataset.filter(lambda x: x[self.cfg.has_label_column_name])

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

        pixel_0_values = self.process_image(example[self.cfg.image_0_column_name])
        pixel_1_values = self.process_image(example[self.cfg.image_1_column_name])

        item = {
            self.cfg.input_ids_column_name: input_ids,
            self.cfg.pixels_0_column_name: pixel_0_values,
            self.cfg.pixels_1_column_name: pixel_1_values,
            self.cfg.label_0_column_name: torch.tensor(example[self.cfg.label_0_column_name])[None],
            self.cfg.label_1_column_name: torch.tensor(example[self.cfg.label_1_column_name])[None],
            self.cfg.num_examples_per_prompt_column_name: torch.tensor(example[self.cfg.num_examples_per_prompt_column_name])[None],
        }
        return item

    def collate_fn(self, batch):
        input_ids = simple_collate(batch, self.cfg.input_ids_column_name)
        pixel_0_values = simple_collate(batch, self.cfg.pixels_0_column_name)
        pixel_1_values = simple_collate(batch, self.cfg.pixels_1_column_name)
        label_0 = simple_collate(batch, self.cfg.label_0_column_name)
        label_1 = simple_collate(batch, self.cfg.label_1_column_name)
        num_examples_per_prompt = simple_collate(batch, self.cfg.num_examples_per_prompt_column_name)

        pixel_0_values = pixel_0_values.to(memory_format=torch.contiguous_format).float()
        pixel_1_values = pixel_1_values.to(memory_format=torch.contiguous_format).float()

        collated = {
            self.cfg.input_ids_column_name: input_ids,
            self.cfg.pixels_0_column_name: pixel_0_values,
            self.cfg.pixels_1_column_name: pixel_1_values,
            self.cfg.label_0_column_name: label_0,
            self.cfg.label_1_column_name: label_1,
            self.cfg.num_examples_per_prompt_column_name: num_examples_per_prompt,
        }
        return collated

    def __len__(self):
        return len(self.dataset)
