"""Evaluate a released model with its original preprocessing and metrics."""
import argparse
import json
from pathlib import Path
from mmengine.config import DictAction
from mmengine.runner import Runner
from reinpp.runtime import read_config, build_model, set_data_root, validate_dataset_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--work-dir', default='work_dirs/eval')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    cfg = read_config(args.config, args.cfg_options)
    cfg.test_dataloader.dataset = set_data_root(cfg.test_dataloader.dataset, args.data_root)
    validate_dataset_paths(cfg.test_dataloader.dataset)
    model = build_model(cfg, args.checkpoint, args.backbone, args.config)
    # Passing an already loaded module prevents Runner checkpoint hooks from
    # overriding trained adapter tensors with the frozen base weights.
    runner = Runner(model=model, work_dir=args.work_dir,
                    test_dataloader=cfg.test_dataloader, test_cfg=cfg.test_cfg,
                    test_evaluator=cfg.test_evaluator, launcher='none',
                    default_scope='mmseg', env_cfg=cfg.get('env_cfg', {}),
                    default_hooks=dict(logger=dict(type='LoggerHook', interval=50)))
    result = runner.test()
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    Path(args.work_dir, 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
