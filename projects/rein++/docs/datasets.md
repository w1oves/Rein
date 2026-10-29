# Evaluation data

Place licensed dataset copies under a common directory supplied as `--data-root`:

| Dataset | Images | Train-ID labels | Filename suffixes (image / label) |
|---|---|---|---|
| Cityscapes | `cityscapes/leftImg8bit/val` | `cityscapes/gtFine/val` | `_leftImg8bit.png` / `_gtFine_labelTrainIds.png` |
| BDD100K | `bdd100k/images/val` | `bdd100k/seg/masks/val` | `.jpg` / `.png` |
| Mapillary Vistas v1.2 | `mapillary_v1.2/validation/images` | `mapillary_v1.2/semseg/val/label` | `.jpg` / `.png` |
| ACDC | `acdc/rgb_anon/CONDITION/val` | `acdc/gt/CONDITION/val` | `_rgb_anon.png` / `_gt_labelTrainIds.png` |

`CONDITION` is `night`, `snow`, `fog`, or `rain`. Preserve the recursive city/sequence subdirectories. BDD100K and ACDC masks must already use Cityscapes-compatible train IDs. Every evaluation mask must contain IDs 0–18 or ignore ID 255.

## Label preparation

For Cityscapes, convert the supplied polygon JSON annotations into a new output directory, then place the resulting city subdirectories under `cityscapes/gtFine/val`. This follows the original experiment's `cityscapesscripts` rasterization. Directly remapping label-ID PNGs can differ at polygon boundaries and is not the recorded preprocessing path:

```bash
python tools/prepare_labels.py cityscapes \
  --labels-dir /absolute/path/original/gtFine/val \
  --output-dir /absolute/path/converted/gtFine/val
```

For Mapillary, use the original v1.2 metadata JSON containing the 66 `labels` entries and indexed annotation PNGs. Directory names in the original download vary; pass the directory that actually contains validation annotation PNGs:

```bash
python tools/prepare_labels.py mapillary \
  --labels-config /absolute/path/original/config_v1.2.json \
  --labels-dir /absolute/path/original/validation/labels \
  --output-dir /absolute/path/data/mapillary_v1.2/semseg/val/label
```

The converter uses the original experiment's class grouping and records the full lookup table in `label-map.json`. Direct Cityscapes class matches are retained. Traffic-sign variants map to traffic sign; on-rails vehicles to train; cyclists and riders to rider; poles and street lights to pole; curbs and pedestrian areas to sidewalk; crosswalks, parking, bike/service lanes, and lane markings to road. Unmatched classes become 255. Do not substitute Mapillary v2.0 metadata.

The released configs use full-resolution Mapillary images and labels from the paths above. The pipeline resizes images while retaining original-resolution ground truth for scoring. Do not substitute the `imageshalf`/`labelhalf` copies or resize annotations during conversion. The conversion commands preserve source files and refuse to overwrite existing output masks.

## Protocol boundaries

Configs preserve original validation transforms, sliding-window crop/stride settings, class order, and domain metrics. Evaluation never loads training datasets. Cityscapes-source generalization configs also evaluate Cityscapes validation; this diagnostic result is separate from the paper's target-domain table. ACDC's four-condition validation mean is not its official test-server result. Train-ID prediction PNGs are not official submission IDs.
