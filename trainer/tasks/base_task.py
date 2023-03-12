from dataclasses import dataclass

from PIL import Image
from accelerate.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BaseTaskConfig:
    limit_examples_to_wandb: int = 50
    pass


class BaseTask:

    def __init__(self, cfg: BaseTaskConfig, accelerator):
        self.accelerator = accelerator
        self.cfg = cfg

    def train_step(self, model, criterion, batch):
        pass

    def valid_step(self, model, criterion, batch):
        pass

    def evaluate(self, model, criterion, dataloader):
        pass

    def log_to_wandb(self, eval_dict):
        if not self.accelerator.is_main_process:
            return
        import wandb
        logger.info("Uploading to wandb")
        for key, value in eval_dict.items():
            eval_dict[key] = [wandb.Image(maybe_img) if isinstance(maybe_img, Image.Image) else maybe_img for maybe_img
                              in value]
            if self.cfg.limit_examples_to_wandb > 0:
                eval_dict[key] = eval_dict[key][:self.cfg.limit_examples_to_wandb]
        columns, predictions = list(zip(*sorted(eval_dict.items())))
        predictions += ([self.accelerator.global_step] * len(predictions[0]),)
        columns += ("global_step",)
        data = list(zip(*predictions))
        table = wandb.Table(columns=list(columns), data=data)
        wandb.log({"test_predictions": table}, commit=False, step=self.accelerator.global_step)
