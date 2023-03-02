from dataclasses import dataclass


@dataclass
class BaseTaskConfig:
    pass


class BaseTask:

    def train_step(self, model, criterion, batch):
        pass

    def valid_step(self, model, criterion, batch):
        pass

    def evaluate(self, model, criterion, dataloader):
        pass

    def log_to_wandb(self, eval_dict):
        import wandb
        logger.info("Uploading to wandb")
        for key, value in eval_dict.items():
            eval_dict[key] = [wandb.Image(maybe_img) if isinstance(maybe_img, Image.Image) else maybe_img for maybe_img
                              in value]
        columns, predictions = list(zip(*sorted(eval_dict.items())))
        predictions += ([self.accelerator.global_step] * len(predictions[0]),)
        columns += ("global_step",)
        data = list(zip(*predictions))
        table = wandb.Table(columns=list(columns), data=data)
        wandb.log({"test_predictions": table}, commit=False, step=self.accelerator.global_step)
