"""Save a 19-class train-ID mask and a colored overlay for one image."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from mmseg.apis import inference_model
from mmseg.datasets import CityscapesDataset
from reinpp.runtime import read_config, build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('image')
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--output-dir', default='outputs')
    args = parser.parse_args()
    if not Path(args.image).is_file():
        raise FileNotFoundError(args.image)
    cfg = read_config(args.config)
    model = build_model(cfg, args.checkpoint, args.backbone, args.config)
    result = inference_model(model, args.image)
    mask = result.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)
    palette = np.asarray(CityscapesDataset.METAINFO['palette'], dtype=np.uint8)
    image = Image.open(args.image).convert('RGB')
    if mask.shape != (image.height, image.width) or mask.max() >= len(palette):
        raise RuntimeError('Unexpected prediction size or class ID')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output / (Path(args.image).stem + '_trainIds.png'))
    Image.blend(image, Image.fromarray(palette[mask]), 0.5).save(output / (Path(args.image).stem + '_overlay.png'))


if __name__ == '__main__':
    main()
