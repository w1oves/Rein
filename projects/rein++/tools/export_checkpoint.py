"""Export a trusted original checkpoint; never publish optimizer or training metadata."""
import argparse
import json
from pathlib import Path
from reinpp.checkpoints import sha256, tensor_state, select_inference_state
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('output')
    parser.add_argument('--branch', required=True, choices=['student', 'teacher', 'teacher_reins'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--trusted-source', action='store_true', help='Allow Python objects in your own original checkpoint; never use for untrusted downloads')
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=not args.trusted_source)
    state = select_inference_state(tensor_state(source), args.branch)
    meta = dict(format_version=1, model_id=args.model_id, branch=args.branch,
                backbone_sha256=sha256(args.backbone), config_sha256=sha256(args.config))
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=state, meta=meta), destination)
    # Source paths and original metadata are intentionally excluded.
    manifest = dict(**meta, file=destination.name, sha256=sha256(destination),
                    bytes=destination.stat().st_size, tensor_count=len(state))
    destination.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
