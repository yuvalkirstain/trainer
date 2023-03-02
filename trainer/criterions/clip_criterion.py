from dataclasses import dataclass

import torch
from torch.nn.modules.loss import _Loss


@dataclass
class CLIPCriterionConfig:
    _target_: str = "trainer.criterions.clip_criterion.CLIPCriterion"
    is_distributed: bool = False
    pass


class CLIPCriterion(_Loss):
    def __init__(self, cfg: CLIPCriterionConfig):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def get_features(model, pixel_values, input_ids, bad_pixel_values):
        all_pixel_values = torch.cat([pixel_values, bad_pixel_values], dim=0)
        all_image_features = model.get_image_features(all_pixel_values)
        text_features = model.get_text_features(input_ids)
        all_image_features = all_image_features / all_image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features, bad_image_features = all_image_features.chunk(2, dim=0)
        return image_features, bad_image_features, text_features

    @staticmethod
    def gather_features(features):
        all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
        return all_features

    def calc_loss(self, image_features, bad_image_features, text_features, logit_scale):
        device = image_features.device
        if self.cfg.is_distributed:
            image_features = self.gather_features(image_features)
            bad_image_features = self.gather_features(bad_image_features)
            text_features = self.gather_features(text_features)
        all_image_features = torch.cat([image_features, bad_image_features], dim=0)  # (2 * batch_size, dim)
        logits_per_image = logit_scale * all_image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ all_image_features.T
        num_logits = logits_per_text.shape[0]
        good_images_logits = logits_per_image.chunk(2, dim=0)[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        image_loss = torch.nn.functional.cross_entropy(good_images_logits, labels)
        text_loss = torch.nn.functional.cross_entropy(logits_per_text, labels)
        loss = (image_loss + text_loss) / 2
        return loss

    def forward(self, model, batch):

        image_features, bad_image_features, text_features = self.get_features(
            model,
            batch["pixel_values"],
            batch["input_ids"],
            batch["bad_pixel_values"]
        )
        loss = self.calc_loss(image_features, bad_image_features, text_features, model.logit_scale.exp())
        return loss
