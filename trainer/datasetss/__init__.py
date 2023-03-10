from hydra.core.config_store import ConfigStore

from trainer.datasetss.clip_hf_dataset import (
    CLIPHFDatasetConfig,
)
from trainer.datasetss.diffusion_hf_dataset import DiffusionHFDatasetConfig

cs = ConfigStore.instance()
cs.store(group="dataset", name="clip", node=CLIPHFDatasetConfig)
cs.store(group="dataset", name="diffusion", node=DiffusionHFDatasetConfig)
