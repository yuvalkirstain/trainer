from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from accelerate.logging import get_logger
from omegaconf import DictConfig
from torch import nn

from trainer.accelerators.base_accelerator import BaseAccelerator
from trainer.configs.configs import TrainerConfig, instantiate_with_cfg

logger = get_logger(__name__)


def load_dataloaders(cfg: DictConfig) -> Any:
    dataloaders = {}
    for split in ["train", "validation", "test"]:
        dataset = instantiate_with_cfg(cfg, split=split)
        should_shuffle = split == "train"
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            shuffle=should_shuffle,
            batch_size=cfg.batch_size,
            collate_fn=dataset.collate_fn,
            num_workers=cfg.num_workers
        )
    return dataloaders


def load_optimizer(cfg: DictConfig, model: nn.Module):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = instantiate(cfg, params=params)
    return optimizer


def load_scheduler(cfg: DictConfig, optimizer):
    scheduler = instantiate_with_cfg(cfg, optimizer=optimizer)
    return scheduler


def load_task(cfg: DictConfig, accelerator: BaseAccelerator):
    task = instantiate_with_cfg(cfg, accelerator=accelerator)
    return task


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: TrainerConfig) -> None:
    accelerator = instantiate_with_cfg(cfg.accelerator)
    task = load_task(cfg.task, accelerator)
    logger.info(f"task: {task.__class__.__name__}")

    model = instantiate_with_cfg(cfg.model)
    logger.info(f"model: {model.__class__.__name__}")
    logger.info(f"num. model params: {int(sum(p.numel() for p in model.parameters()) // 1e6)}M")
    logger.info(f"num. model trainable params: {int(sum(p.numel() for p in model.parameters() if p.requires_grad) // 1e6)}M")

    criterion = instantiate_with_cfg(cfg.criterion)
    logger.info(f"criterion: {criterion.__class__.__name__}")

    dataloaders = load_dataloaders(cfg.dataset)

    optimizer = load_optimizer(cfg.optimizer, model)
    lr_scheduler = load_scheduler(cfg.lr_scheduler, optimizer)

    model, optimizer, lr_scheduler, dataloaders = accelerator.prepare(model, optimizer, lr_scheduler, dataloaders)

    accelerator.load_state_if_needed()

    accelerator.recalc_train_length_after_prepare(len(dataloaders["train"]))

    if accelerator.is_main_process:
        accelerator.init_training(cfg)

    for epoch in range(accelerator.cfg.num_epochs):
        train_loss = 0.0
        for step, batch in enumerate(dataloaders["train"]):
            if accelerator.should_skip(epoch, step):
                accelerator.update_progbar_step()
                continue

            if accelerator.should_eval():
                model.eval()
                metrics = task.evaluate(model, criterion, dataloaders["validation"])
                accelerator.update_metrics(metrics)

            model.train()

            with accelerator.accumulate(model):
                loss = task.train_step(model, criterion, batch)
                avg_loss = accelerator.gather(loss).mean().item()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters())

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            train_loss += avg_loss / accelerator.cfg.gradient_accumulation_steps

            if accelerator.sync_gradients:
                accelerator.update_global_step(train_loss)
                train_loss = 0.0

            accelerator.update_step(avg_loss, lr_scheduler.get_last_lr()[0])

            if accelerator.should_save():
                accelerator.save_checkpoint()

            if accelerator.should_end():
                break

        if accelerator.should_end():
            break

        accelerator.update_epoch()

    accelerator.load_best_checkpoint()
    task.evaluate(model, criterion, dataloaders["test"])
    accelerator.save_unwrapped_model()
    accelerator.end_training()


if __name__ == '__main__':
    main()
