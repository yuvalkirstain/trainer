import collections
from dataclasses import dataclass

import torch
from PIL import Image
from accelerate.logging import get_logger
from accelerate.utils import LoggerType, PrecisionType
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
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
class InferenceNoiseSchedulerConfig:
    _target_: str = "diffusers.DPMSolverMultistepScheduler.from_pretrained"
    pretrained_model_name_or_path: str = II("model.pretrained_model_name_or_path")
    subfolder: str = "scheduler"


@dataclass
class DiffusionTaskConfig(BaseTaskConfig):
    _target_: str = "trainer.tasks.diffusion_task.DiffusionTask"
    pretrained_model_name_or_path: str = II("model.pretrained_model_name_or_path")
    classifier_free_ratio: float = 0.1
    noise_scheduler_cfg: InferenceNoiseSchedulerConfig = InferenceNoiseSchedulerConfig()
    guidance_scale: float = 7.5
    num_inference_steps: int = 25
    max_batches_to_generate: int = 1


class DiffusionTask(BaseTask):
    def __init__(self, cfg: DiffusionTaskConfig, accelerator: BaseAccelerator):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="tokenizer")
        self.accelerator = accelerator
        self.cfg = cfg
        self.unconditional_input_ids = self.init_unconditional_input_ids()
        self.noise_scheduler = instantiate(self.cfg.noise_scheduler_cfg)

    def init_unconditional_input_ids(self):
        return self.tokenizer(
            "",
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

    def train_step(self, model, criterion, batch):
        if self.unconditional_input_ids is None:
            self.init_unconditional_input_ids(model)
        input_ids, pixel_values = batch["input_ids"], batch["pixel_values"]
        classifier_free_mask = torch.rand(size=(input_ids.shape[0],)) < self.cfg.classifier_free_ratio
        input_ids[classifier_free_mask] = self.unconditional_input_ids.to(input_ids.device)
        loss = criterion(model, batch)
        return loss

    @torch.no_grad()
    def valid_step(self, model, criterion, batch):
        loss = criterion(model, batch)
        return loss

    def run_inference(self, model, criterion, dataloader):
        eval_dict = collections.defaultdict(list)
        generator = torch.Generator(device=self.accelerator.device)
        generator.manual_seed(self.accelerator.cfg.seed)
        for i, batch in enumerate(dataloader):
            if i >= self.cfg.max_batches_to_generate:
                break
            loss = self.valid_step(model, criterion, batch)
            # TODO add reduction to loss
            eval_dict["loss"] += [loss.item()] * batch["input_ids"].shape[0]
            batch_captions = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            batch_images = model.pipeline(batch_captions,
                                          guidance_scale=self.cfg.guidance_scale,
                                          generator=generator,
                                          num_inference_steps=self.cfg.num_inference_steps).images
            eval_dict["captions"] += batch_captions
            eval_dict["images"] += batch_images
        return eval_dict

    def evaluate(self, model, criterion, dataloader):
        eval_dict = self.run_inference(model, criterion, dataloader)
        if LoggerType.WANDB == self.accelerator.cfg.log_with and self.accelerator.is_main_process:
            self.log_to_wandb(eval_dict)
        metrics = {"loss": sum(eval_dict["loss"]) / len(eval_dict["loss"])}
        return metrics
