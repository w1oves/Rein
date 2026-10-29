"""Convert evaluation annotations into Cityscapes train IDs, without resizing."""
import argparse
import json
from pathlib import Path
from PIL import Image
from reinpp.labels import mapillary_lookup, convert_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['cityscapes', 'mapillary'])
    parser.add_argument('--labels-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--labels-config', help='Mapillary Vistas v1.2 JSON metadata')
    args = parser.parse_args()
    source, output = Path(args.labels_dir).resolve(), Path(args.output_dir).resolve()
    if source == output:
        raise ValueError('Use a separate output directory to preserve original labels')
    if args.dataset == 'mapillary':
        if not args.labels_config:
            parser.error('--labels-config is required for Mapillary')
        labels = json.loads(Path(args.labels_config).read_text())['labels']
        lookup = mapillary_lookup(labels)
        pattern = '*.png'
    else:
        from cityscapesscripts.preparation.json2labelImg import json2labelImg
        lookup = None
        pattern = '*_gtFine_polygons.json'
    files = sorted(source.rglob(pattern))
    if not files:
        raise FileNotFoundError(f'No annotation files matching {pattern} in {source}')
    targets = []
    for path in files:
        relative = path.relative_to(source)
        if args.dataset == 'cityscapes':
            relative = relative.with_name(relative.name.replace('_polygons.json', '_labelTrainIds.png'))
        target = output / relative
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite {target}')
        targets.append((path, target))
    output.mkdir(parents=True, exist_ok=True)
    for path, target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        if args.dataset == 'cityscapes':
            json2labelImg(str(path), str(target), 'trainIds')
        else:
            with Image.open(path) as mask:
                converted = convert_mask(mask, lookup)
            Image.fromarray(converted).save(target)
    (output/'label-map.json').write_text(json.dumps(dict(dataset=args.dataset,
        source_to_train_id=lookup.tolist() if lookup is not None else 'cityscapesscripts polygon trainIds',
        ignore_index=255,resized=False),indent=2)+'\n')
    print(f'Converted {len(files)} masks to {output}')


if __name__ == '__main__':
    main()
