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


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def gather_iterable(it, num_processes):
    output_objects = [None for _ in range(num_processes)]
    torch.distributed.all_gather_object(output_objects, it)
    return flatten(output_objects)


class CLIPTask(BaseTask):
    def __init__(self, cfg: CLIPTaskConfig, accelerator: BaseAccelerator):
        super().__init__(cfg, accelerator)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name_or_path)

    def train_step(self, model, criterion, batch):
        loss = criterion(model, batch)
        return loss

    @torch.no_grad()
    def valid_step(self, model, criterion, batch):
        input_ids, pixel_values, bad_pixel_values = batch["input_ids"], batch["pixel_values"], batch["bad_pixel_values"]
        image_features, bad_image_features, text_features = criterion.get_features(
            model,
            pixel_values,
            input_ids,
            bad_pixel_values
        )
        clip_scores = model.logit_scale.exp() * torch.diag(torch.einsum('bd,cd->bc', text_features, image_features))
        bad_clip_scores = model.logit_scale.exp() * torch.diag(torch.einsum('bd,cd->bc', text_features, bad_image_features))
        scores = torch.stack([clip_scores, bad_clip_scores], dim=-1)
        probs = torch.softmax(scores, dim=-1)
        clip_probes, bad_clip_probes = probs[:, 0], probs[:, 1]
        return clip_probes, bad_clip_probes

    def run_clip_score(self, model, criterion, dataloader):
        eval_dict = collections.defaultdict(list)
        logger.info("Running clip score...")
        for batch in dataloader:
            input_ids, pixel_values, bad_pixel_values = batch["input_ids"], batch["pixel_values"], batch["bad_pixel_values"]
            clip_probs, bad_clip_probs = self.valid_step(model, criterion, batch)
            eval_dict["is_correct"] += (clip_probs > bad_clip_probs).tolist()
            eval_dict["captions"] += self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            eval_dict["images"] += pixel_values_to_pil_images(pixel_values)
            eval_dict["bad_images"] += pixel_values_to_pil_images(bad_pixel_values)
            eval_dict["good_probs"] += clip_probs.tolist()
            eval_dict["bad_probs"] += bad_clip_probs.tolist()

        logger.info("Gathering eval results from all processes...")
        for k, v in eval_dict.items():
            eval_dict[k] = gather_iterable(v, self.accelerator.num_processes)

        return eval_dict

    @torch.no_grad()
    def evaluate(self, model, criterion, dataloader):
        eval_dict = self.run_clip_score(model, criterion, dataloader)
        metrics = {
            "accuracy": sum(eval_dict["is_correct"]) / len(eval_dict["is_correct"]),
            "num_samples": len(eval_dict["is_correct"])
        }
        if LoggerType.WANDB == self.accelerator.cfg.log_with:
            self.log_to_wandb(eval_dict)
        return metrics
