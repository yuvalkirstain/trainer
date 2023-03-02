from dataclasses import dataclass

import torch
from diffusers import StableDiffusionPipeline
from torch import nn

from trainer.models.base_model import BaseModelConfig


@dataclass
class DiffusionModelConfig(BaseModelConfig):
    _target_: str = "trainer.models.diffusion_model.DiffusionModel"
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"


class DiffusionModel(nn.Module):
    def __init__(self, cfg: DiffusionModelConfig):
        super().__init__()
        self.pipeline = StableDiffusionPipeline.from_pretrained(cfg.pretrained_model_name_or_path)
        self.pipeline.set_progress_bar_config(disable=True)
        self.unet = self.pipeline.unet
        self.text_encoder = self.pipeline.text_encoder

    def forward(self, noisy_latents, input_ids, timesteps):
        encoder_hidden_states = self.text_encoder(input_ids)[0]
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return model_pred


