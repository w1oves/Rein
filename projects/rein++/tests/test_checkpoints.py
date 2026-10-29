import pytest
import torch
from reinpp.checkpoints import merge_states, select_inference_state, load_weights, sha256


def test_adapter_wins_over_base():
    expected = {'backbone.w': torch.zeros(2), 'backbone.reins.x': torch.zeros(2), 'decode_head.w': torch.zeros(1)}
    base = {'w': torch.ones(2), 'reins.x': torch.zeros(2)}
    trained = {'backbone.reins.x': torch.full((2,), 7.), 'decode_head.w': torch.ones(1)}
    result = merge_states(expected, base, trained)
    assert torch.equal(result['backbone.reins.x'], trained['backbone.reins.x'])


@pytest.mark.parametrize('change', ['missing', 'unexpected', 'shape'])
def test_incompatible_weights_rejected(change):
    expected = {'backbone.w': torch.zeros(2), 'decode_head.w': torch.zeros(1)}
    base = {'w': torch.ones(2)}
    adapter = {'decode_head.w': torch.ones(1)}
    if change == 'missing':
        base = {}
    elif change == 'unexpected':
        adapter['teachers.hidden'] = torch.ones(1)
    else:
        base['w'] = torch.zeros(3)
    with pytest.raises(ValueError, match='Incompatible weights'):
        merge_states(expected, base, adapter)


@pytest.mark.parametrize('branch', ['student', 'teacher', 'teacher_reins'])
def test_export_branch_is_explicit_and_training_state_removed(branch):
    state = {'backbone.reins.x': torch.tensor([1.]), 'decode_head.w': torch.tensor([1.]),
             'teachers.backbone.reins.x': torch.tensor([2.]), 'teachers.decode_head.w': torch.tensor([2.]),
             'teachers.reins.x': torch.tensor([3.]), 'ta_head.w': torch.tensor([4.]),
             'backbone.frozen': torch.tensor([5.])}
    selected = select_inference_state(state, branch)
    assert set(selected) == {'backbone.reins.x', 'decode_head.w'}
    assert selected['backbone.reins.x'].item() == {'student': 1, 'teacher': 2, 'teacher_reins': 3}[branch]


def test_missing_teacher_is_not_silently_replaced_with_student():
    with pytest.raises(ValueError, match='No Rein adapter'):
        select_inference_state({'backbone.reins.x': torch.ones(1)}, 'teacher')


def test_load_checks_hash_before_mutating_model(tmp_path):
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(2, 2)
    original = model.backbone.weight.detach().clone()
    base = tmp_path / 'base.pth'
    torch.save(model.backbone.state_dict(), base)
    checkpoint = tmp_path / 'adapter.pth'
    torch.save({'state_dict': {'a': torch.ones(1)}, 'meta': {'format_version': 1, 'backbone_sha256': '0' * 64}}, checkpoint)
    with pytest.raises(ValueError, match='SHA-256'):
        load_weights(model, checkpoint, base)
    assert torch.equal(model.backbone.weight, original)


def test_missing_file_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='Weight file not found'):
        load_weights(torch.nn.Linear(1, 1), tmp_path / 'missing.pth', tmp_path / 'base.pth')
