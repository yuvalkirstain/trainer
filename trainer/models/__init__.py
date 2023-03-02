from hydra.core.config_store import ConfigStore

from trainer.models.clip_model import ClipModelConfig
from trainer.models.diffusion_model import DiffusionModelConfig

cs = ConfigStore.instance()
cs.store(group="model", name="clip", node=ClipModelConfig)
cs.store(group="model", name="diffusion", node=DiffusionModelConfig)

