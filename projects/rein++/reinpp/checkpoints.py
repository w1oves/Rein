"""Tensor-only checkpoint export and strict, base-first loading."""
from collections.abc import Mapping
from pathlib import Path
import hashlib
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_state(checkpoint):
    state = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, Mapping) else None
    if not isinstance(state, Mapping) or not state:
        raise ValueError('Checkpoint does not contain a nonempty state dictionary')
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError('State dictionary must contain only named tensors')
    return dict(state)


def select_inference_state(state, branch):
    """Select an explicit inference branch; never mix teacher and student weights."""
    if branch not in ('student', 'teacher', 'teacher_reins'):
        raise ValueError('Specify student, teacher, or teacher_reins')
    if state and all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    selected = {}
    for key, value in state.items():
        target = key
        if branch == 'teacher':
            if not key.startswith('teachers.'):
                continue
            target = key[len('teachers.'):]
        elif branch == 'teacher_reins':
            if key.startswith('teachers.reins.'):
                target = 'backbone.reins.' + key[len('teachers.reins.'):]
            elif key.startswith('teachers.decode_head.'):
                target = key[len('teachers.'):]
            else:
                continue
        if target.startswith(('backbone.reins.', 'decode_head.', 'neck.')):
            if target in selected:
                raise ValueError(f'Duplicate exported tensor: {target}')
            selected[target] = value.detach().cpu().clone()
    if not any(k.startswith('backbone.reins.') for k in selected):
        raise ValueError(f'No Rein adapter parameters found for branch {branch}')
    if not any(k.startswith('decode_head.') for k in selected):
        raise ValueError(f'No segmentation head found for branch {branch}')
    return selected


def merge_states(expected, base, adapter):
    """Validate all keys and shapes before mutating the model."""
    base = {k if k.startswith('backbone.') else 'backbone.' + k: v for k, v in base.items()}
    unexpected = sorted((base.keys() | adapter.keys()) - expected.keys())
    merged = {**base, **adapter}
    missing = sorted(expected.keys() - merged.keys())
    wrong = [k for k in expected.keys() & merged.keys() if expected[k].shape != merged[k].shape]
    if unexpected or missing or wrong:
        raise ValueError(f'Incompatible weights: missing={missing[:20]}, unexpected={unexpected[:20]}, shape_mismatch={wrong[:20]}')
    return merged


def load_weights(model, checkpoint, backbone, config=None):
    for path in (checkpoint, backbone):
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f'Weight file not found: {path}')
    published = torch.load(checkpoint, map_location='cpu', weights_only=True)
    meta = published.get('meta', {})
    if meta.get('format_version') != 1:
        raise ValueError('Use a Rein++ inference export (format_version=1)')
    expected_hash = meta.get('backbone_sha256')
    if not expected_hash or sha256(backbone) != expected_hash:
        raise ValueError('Base-model SHA-256 does not match this checkpoint')
    if config and sha256(config) != meta.get('config_sha256'):
        raise ValueError('Config SHA-256 does not match this checkpoint')
    base = tensor_state(torch.load(str(backbone), map_location='cpu', weights_only=True, mmap=True))
    adapter = tensor_state(published)
    state = merge_states(model.state_dict(), base, adapter)
    model.load_state_dict(state, strict=True)
    return meta
