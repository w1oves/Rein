from pathlib import Path
from copy import deepcopy
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmseg.registry import MODELS
from . import register_modules
from .checkpoints import load_weights


def read_config(path, options=None):
    cfg = Config.fromfile(path)
    if options:
        cfg.merge_from_dict(options)
    if any(cfg.get(k) is not None for k in ('train_dataloader', 'train_cfg', 'optim_wrapper')):
        raise ValueError('Use an inference-only release config')
    register_modules()
    init_default_scope('mmseg')
    return cfg


def build_model(cfg, checkpoint, backbone, config_path, device='cuda:0'):
    import torch
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use a GPU environment with MMCV CUDA operators')
    model = MODELS.build(deepcopy(cfg.model))
    load_weights(model, checkpoint, backbone, config=config_path)
    model.cfg = cfg
    return model.to(device).eval()


def set_data_root(dataset, root):
    """Relocate only datasets; never silently change preprocessing or labels."""
    dataset = deepcopy(dataset)
    if 'datasets' in dataset:
        dataset['datasets'] = [set_data_root(d, root) for d in dataset['datasets']]
    elif 'dataset' in dataset:
        dataset['dataset'] = set_data_root(dataset['dataset'], root)
    elif 'data_root' in dataset:
        old = Path(dataset['data_root'])
        if old.is_absolute() or '..' in old.parts:
            raise ValueError(f'Expected a portable dataset root, got {old}')
        relative = Path(*old.parts[1:]) if old.parts and old.parts[0] == 'data' else old
        dataset['data_root'] = str(Path(root).resolve() / relative)
    return dataset


def validate_dataset_paths(dataset):
    if 'datasets' in dataset:
        for child in dataset['datasets']:
            validate_dataset_paths(child)
        return
    if 'dataset' in dataset:
        validate_dataset_paths(dataset['dataset'])
        return
    root = Path(dataset.get('data_root', ''))
    for key, relative in dataset.get('data_prefix', {}).items():
        path = root / relative
        if not path.is_dir():
            raise FileNotFoundError(f'Dataset {key} directory does not exist: {path}')
