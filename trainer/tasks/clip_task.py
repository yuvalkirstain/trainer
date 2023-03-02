import collections
from dataclasses import dataclass

import torch
from PIL import Image
from accelerate.logging import get_logger
from accelerate.utils import LoggerType
from hydra.core.config_store import ConfigStore
from omegaconf import II
from transformers import AutoTokenizer

from trainer.accelerators.base_accelerator import BaseAccelerator
from trainer.tasks.base_task import BaseTaskConfig, BaseTask

logger = get_logger(__name__)


def numpy_to_pil(images):
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def pixel_values_to_pil_images(pixel_values):
    images = (pixel_values / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    images = numpy_to_pil(images)
    return images


@dataclass
class CLIPTaskConfig(BaseTaskConfig):
    _target_: str = "trainer.tasks.clip_task.CLIPTask"
    pretrained_model_name_or_path: str = II("model.pretrained_model_name_or_path")


class CLIPTask(BaseTask):
    def __init__(self, cfg: CLIPTaskConfig, accelerator: BaseAccelerator):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name_or_path)
        self.accelerator = accelerator

    def train_step(self, model, criterion, batch):
        loss = criterion(model, batch)
        return loss

    def valid_step(self, model, criterion, batch):
        input_ids, pixel_values, bad_pixel_values = batch["input_ids"], batch["pixel_values"], batch["bad_pixel_values"]
        image_features, bad_image_features, text_features = criterion.get_features(
            model,
            pixel_values,
            input_ids,
            bad_pixel_values
        )
        clip_scores = torch.diag(torch.einsum('bd,cd->bc', text_features, image_features))
        bad_clip_scores = torch.diag(torch.einsum('bd,cd->bc', text_features, bad_image_features))
        return clip_scores, bad_clip_scores

    def run_clip_score(self, model, criterion, dataloader):
        eval_dict = collections.defaultdict(list)
        for batch in dataloader:
            input_ids, pixel_values, bad_pixel_values = batch["input_ids"], batch["pixel_values"], batch[
                "bad_pixel_values"]
            clip_scores, bad_clip_scores = self.valid_step(model, criterion, batch)
            eval_dict["is_correct"] += (clip_scores > bad_clip_scores).tolist()
            eval_dict["captions"] += self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            eval_dict["images"] += pixel_values_to_pil_images(pixel_values)
            eval_dict["bad_images"] += pixel_values_to_pil_images(bad_pixel_values)
            eval_dict["scores"] += clip_scores.tolist()
            eval_dict["bad_scores"] += bad_clip_scores.tolist()
        return eval_dict

    def evaluate(self, model, criterion, dataloader):
        eval_dict = self.run_clip_score(model, criterion, dataloader)
        if LoggerType.WANDB == self.accelerator.cfg.log_with and self.accelerator.is_main_process:
            self.log_to_wandb(eval_dict)
        metrics = {"accuracy": sum(eval_dict["is_correct"]) / len(eval_dict["is_correct"])}
        return metrics
