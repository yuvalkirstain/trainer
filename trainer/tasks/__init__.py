
from hydra.core.config_store import ConfigStore

from trainer.tasks.clip_task import CLIPTaskConfig
from trainer.tasks.diffusion_task import DiffusionTaskConfig

cs = ConfigStore.instance()
cs.store(group="task", name="clip", node=CLIPTaskConfig)
cs.store(group="task", name="diffusion", node=DiffusionTaskConfig)

