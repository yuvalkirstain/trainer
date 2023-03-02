from hydra.core.config_store import ConfigStore
from trainer.criterions.clip_criterion import CLIPCriterionConfig
from trainer.criterions.diffusion_criterion import DiffusionCriterionConfig

cs = ConfigStore.instance()
cs.store(group="criterion", name="clip", node=CLIPCriterionConfig)
cs.store(group="criterion", name="diffusion", node=DiffusionCriterionConfig)
