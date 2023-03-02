import abc
import hashlib
import json
import math
import os
from dataclasses import field, dataclass
from glob import glob
from typing import List, Optional

import datasets
import diffusers
import torch
import transformers
from accelerate.logging import get_logger
from accelerate.utils import set_seed as accelerate_set_seed, PrecisionType
from accelerate.utils.dataclasses import BaseEnum, LoggerType
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from trainer.accelerators.utils import get_nvidia_smi_gpu_memory_stats_str, print_config

logger = get_logger(__name__)


def debug(port):
    logger.info("Connecting to debugger...")
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=port, stdoutToServer=True, stderrToServer=True)


@dataclass
class DebugConfig:
    activate: bool = False
    port: int = 5900


class TrainingMode(BaseEnum):
    SKIPPING = "skipping"
    TRAINING = "training"


@dataclass
class BaseAcceleratorConfig:
    _target_: str = "trainer.accelerators.base_accelerator.Accelerator"
    output_dir: str = "output"
    mixed_precision: PrecisionType = PrecisionType.NO
    gradient_accumulation_steps: int = 1
    log_with: Optional[LoggerType] = LoggerType.WANDB
    debug: DebugConfig = DebugConfig()
    seed: int = 42
    resume_from_checkpoint: bool = False
    max_steps: int = 1000
    num_epochs: int = 10
    validate_steps: int = 100
    eval_on_start: bool = False
    project_name: str = "reward"
    max_grad_norm: float = 1.0
    save_steps: int = 100


class BaseAccelerator(abc.ABC):

    def __init__(self, cfg: BaseAcceleratorConfig):
        self.cfg = cfg
        self.accelerator = None
        self.epoch = 0
        self.step = 0
        self.global_step = 0
        self.step_loss = 0.0
        self.lr = None
        self.metrics = {}
        self.progress_bar = None
        self.mode = TrainingMode.TRAINING

    def post_init(self):
        self.set_seed()
        self.debug()
        self.set_logging_level()

    def set_logging_level(self):
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

    def debug(self):
        if self.accelerator.is_main_process and self.cfg.debug.activate:
            debug(self.cfg.debug.port)

    def set_seed(self):
        logger.info(f"Setting seed {self.cfg.seed}")
        accelerate_set_seed(self.cfg.seed, device_specific=True)

    def prepare(self, *args, device_placement=None):
        return self.accelerator.prepare(*args, device_placement=device_placement)

    def get_latest_checkpoint(self):
        all_ckpts = list(glob(os.path.join(self.cfg.output_dir, "checkpoint-*")))
        all_ckpts.sort(key=os.path.getctime)
        if len(all_ckpts) > 0 and "final" in all_ckpts[-1]:
            all_ckpts.pop()
            return all_ckpts[-1]

    def load_state_if_needed(self):
        if not self.cfg.resume_from_checkpoint:
            return
        ckpt_path = self.get_latest_checkpoint()

        if ckpt_path is None:
            logger.info("No checkpoint found, training from scratch")
            return

        stage = json.load(open(os.path.join(ckpt_path, "training_stage.json")))
        self.epoch, self.step, self.global_step, self.metrics = stage["epoch"], stage["step"], stage["global_step"], stage["metrics"]
        logger.info(
            f"Resuming from checkpoint: {ckpt_path} | epoch={self.epoch} step={self.step} gstep={self.global_step}")
        self.accelerator.load_state(ckpt_path)
        logger.info("Checkpoint loaded")

    @property
    def is_main_process(self):
        return self.accelerator.is_main_process

    def pre_training_log(self, cfg: DictConfig):
        total_batch_size = cfg.dataset.batch_size * self.accelerator.num_processes * self.cfg.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Instantaneous batch size per device = {cfg.dataset.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.cfg.gradient_accumulation_steps}")
        logger.info(f"  Total warmup steps = {cfg.lr_scheduler.lr_warmup_steps}")
        logger.info(f"  Total training steps = {self.cfg.max_steps}")
        logger.info(f"  Total optimization steps = {self.cfg.max_steps}")
        logger.info(f"  Mixed precision = {self.cfg.mixed_precision}")

    def init_training(self, cfg: DictConfig):
        self.accelerator.init_trackers(self.cfg.project_name, config=OmegaConf.to_object(cfg))
        logger.info(get_nvidia_smi_gpu_memory_stats_str())
        print_config(cfg)
        self.pre_training_log(cfg)
        self.progress_bar = tqdm(range(self.cfg.max_steps), disable=not self.accelerator.is_main_process)
        self.progress_bar.set_description("Steps")

    def should_skip(self, epoch, step):
        should = epoch < self.epoch or (epoch == self.epoch and step < self.step)
        if should:
            self.mode = TrainingMode.SKIPPING
            self.progress_bar.set_postfix(**{"status": TrainingMode.SKIPPING})
        else:
            self.mode = TrainingMode.TRAINING
        return should

    def update_progbar_step(self):
        self.progress_bar.update(1)

    def should_eval(self):
        if not self.mode == TrainingMode.TRAINING:
            return False
        if self.step == 0 and self.global_step == 0 and self.cfg.eval_on_start:
            return True
        if self.accelerator.sync_gradients and self.global_step % self.cfg.validate_steps == 0:
            return True
        return False

    def log(self, data):
        self.accelerator.log(data, step=self.global_step)

    def recalc_train_length_after_prepare(self, num_examples):
        num_update_steps_per_epoch = math.ceil(num_examples / self.cfg.gradient_accumulation_steps)
        if self.cfg.max_steps is None:
            self.cfg.max_steps = self.cfg.num_epochs * num_update_steps_per_epoch
        self.cfg.num_epochs = math.ceil(self.cfg.max_steps / num_update_steps_per_epoch)

    def accumulate(self, model):
        return self.accelerator.accumulate(model)

    def gather(self, data):
        return self.accelerator.gather(data)

    @property
    def sync_gradients(self):
        return self.accelerator.sync_gradients

    def update_step_loss(self, loss):
        self.step_loss = loss

    def update_global_step(self, loss):
        self.global_step += 1
        self.log({
            "lr": self.lr,
            "step": self.step,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "loss": loss,
        })

    def get_allocated_cuda_memory(self):
        return round(torch.cuda.max_memory_allocated(self.accelerator.device) / 1024 / 1024 / 1024, 2)

    def update_step(self, loss, lr):
        self.step += 1
        self.lr = lr
        logs = {
            "stl": loss,
            "gstl": loss,
            "mem": self.get_allocated_cuda_memory(),
            "st": self.step,
            "ep": self.epoch,
            "gst": self.global_step,
            "lr": self.lr,
        }
        self.progress_bar.set_postfix(**logs)
        self.update_progbar_step()

    def update_epoch(self):
        if self.mode == TrainingMode.SKIPPING:
            return
        self.epoch += 1
        self.step = 0

    def update_metrics(self, metrics):
        self.metrics.update(metrics)
        self.log(metrics)

    def should_end(self):
        return self.global_step >= self.cfg.max_steps

    def backward(self, loss):
        self.accelerator.backward(loss)

    def clip_grad_norm_(self, params):
        self.accelerator.clip_grad_norm_(params, self.cfg.max_grad_norm)

    def should_save(self):
        return self.sync_gradients and self.global_step > 0 and self.cfg.save_steps > 0 and self.global_step % self.cfg.save_steps == 0

    def save_checkpoint(self):
        training_stage = {
            "epoch": self.epoch,
            "step": self.step,
            "global_step": self.global_step,
            "step_loss": self.step_loss,
            "lr": self.lr,
            "metrics": self.metrics,
        }
        save_hash = hashlib.md5(json.dumps(training_stage, sort_keys=True).encode('utf-8')).hexdigest()
        save_dir = os.path.join(self.cfg.output_dir, f"checkpoint-{save_hash}")
        logger.info(f"Saving checkpoint to {save_dir}")
        json.dump(training_stage, open(os.path.join(save_dir, "training_stage.json"), "w"), indent=4)
        self.accelerator.save_state(save_dir)
        logger.info(f"Saved checkpoint to {save_dir}")

    @property
    def device(self):
        return self.accelerator.device
