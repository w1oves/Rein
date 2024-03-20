from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.runner.checkpoint import _load_checkpoint


@HOOKS.register_module()
class LoadBackboneHook(Hook):
    def __init__(self, checkpoint_path) -> None:
        self.checkpoint_path = checkpoint_path

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        converted_backbone_weight = _load_checkpoint(
            self.checkpoint_path, map_location="cpu"
        )
        if "state_dict" in checkpoint:
            checkpoint["state_dict"].update(
                {f"backbone.{k}": v for k, v in converted_backbone_weight.items()}
            )
        else:
            checkpoint.update(
                {f"backbone.{k}": v for k, v in converted_backbone_weight.items()}
            )
