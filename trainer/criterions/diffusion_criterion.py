from dataclasses import dataclass
import torch
from accelerate.utils import PrecisionType
from hydra.utils import instantiate
from omegaconf import II
from torch.nn.modules.loss import _Loss


@dataclass
class TrainNoiseSchedulerConfig:
    _target_: str = "diffusers.DDPMScheduler.from_pretrained"
    pretrained_model_name_or_path: str = II("model.pretrained_model_name_or_path")
    subfolder: str = "scheduler"


@dataclass
class DiffusionCriterionConfig:
    _target_: str = "trainer.criterions.diffusion_criterion.DiffusionCriterion"
    noise_scheduler_cfg: TrainNoiseSchedulerConfig = TrainNoiseSchedulerConfig()
    mixed_precision: PrecisionType = II("accelerator.mixed_precision")


class DiffusionCriterion(_Loss):
    def __init__(self, cfg: DiffusionCriterionConfig):
        super().__init__()
        self.cfg = cfg
        self.noise_scheduler = instantiate(self.cfg.noise_scheduler_cfg)
        self.weight_dtype = self.infer_dtype()

    def infer_dtype(self):
        if self.cfg.mixed_precision == PrecisionType.FP16:
            return torch.float16
        elif self.cfg.mixed_precision == PrecisionType.BF16:
            return torch.bfloat16
        return torch.float32

    @torch.no_grad()
    def encode_image(self, model, image):
        latents = model.pipeline.vae.encode(image.to(self.weight_dtype)).latent_dist.sample()
        latents = latents * 0.18215
        return latents

    def forward(self, model, batch):
        input_ids, pixel_values = batch["input_ids"], batch["pixel_values"]

        with torch.no_grad():
            latents = self.encode_image(model, pixel_values).float()

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (latents.shape[0],),
                                  device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        target = None
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        noisy_latents = noisy_latents.to(self.weight_dtype)
        model_pred = model(noisy_latents, input_ids, timesteps)
        loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss
