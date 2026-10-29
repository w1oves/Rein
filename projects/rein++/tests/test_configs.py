from pathlib import Path
from mmengine import Config
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('path', sorted((ROOT / 'configs').glob('*.py')))
def test_configs_are_inference_only_and_self_contained(path):
    cfg = Config.fromfile(path)
    assert 'train_dataloader' not in cfg
    assert 'optim_wrapper' not in cfg
    assert cfg.model.train_cfg is None
    assert cfg.model.type == 'EncoderDecoder'
    assert 'uda_cfg' not in cfg.model and 'ta_head' not in cfg.model
    assert not any(x in path.read_text() for x in ['/mnt/', '/root/', 'load_from'])
    assert cfg.test_dataloader.batch_size == 1
    assert cfg.model.decode_head.num_classes == 19
    assert all(t.type != 'LoadAnnotations' for t in cfg.test_pipeline)
