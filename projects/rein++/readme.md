# Rein++

Official code for **Rein++: Efficient Generalization and Adaptation for Semantic Segmentation With Vision Foundation Models** (IEEE TPAMI 2026).

[Paper](https://doi.org/10.1109/TPAMI.2026.3681631) · [Checkpoints](https://huggingface.co/zhixiangwei/ReinPlusPlus) · [Models and results](docs/models.md) · [Validation report](docs/validation.md)

## First release

We release the first batch of Rein++ checkpoints and inference and evaluation code for reproducing our paper results. This release includes five Rein-G models based on DINOv3-L and DINOv3-7B, and one Rein-A model, covering experiments in Tables II and VII. All six models have been evaluated on their corresponding validation sets. The remaining checkpoints and training code will be released progressively.

This subproject uses the independent `reinpp` Python package. The CVPR 2024 Rein implementation remains in the repository root.

## Installation

Run the following commands from `projects/rein++` in a Linux environment with Python 3.10 and an NVIDIA GPU:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==1.26.4
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Install MMCV 2.1.0 with CUDA operators. For GPUs supported by the OpenMMLab CUDA 11.8 wheels:

```bash
python -m pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
```

Set `CUDA_HOME`, `CC`, and `CXX` to the matching toolkit and compiler if needed. See the [MMCV build instructions](https://github.com/open-mmlab/mmcv/blob/v2.1.0/docs/en/get_started/build.md). The package requires the CUDA operators provided by `mmcv`.

## Models and base weights

The [model table](docs/models.md) links each paper experiment to its configuration and checkpoint. Each Hugging Face model folder contains `model.pth`, `config.py`, and `manifest.json`.

Released checkpoints contain the Rein adapter parameters and segmentation head. Obtain the matching original PyTorch LVD-1689M pretrained backbone from [DINOv3's official repository](https://github.com/facebookresearch/dinov3#pretrained-models):

- DINOv3-L: `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`.
- DINOv3-7B: `dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth`.

Use these original PyTorch files directly. The loader checks the backbone and configuration SHA-256 values, loads the base model, and then applies the adapter and segmentation-head parameters. Complete checksums are recorded in [model-index.json](model-index.json) and each weight manifest.

## Download

Download one model from [Hugging Face](https://huggingface.co/zhixiangwei/ReinPlusPlus):

```bash
python -m pip install huggingface_hub
hf download zhixiangwei/ReinPlusPlus \
  rdg_usg2cbm_dinov3_rein/model.pth \
  rdg_usg2cbm_dinov3_rein/config.py \
  rdg_usg2cbm_dinov3_rein/manifest.json \
  --local-dir /absolute/path/reinpp-weights
```

Replace the model ID with another entry in the model table as needed. Use its matching configuration and backbone in the commands below.

## Single-image inference

```bash
python tools/infer.py /absolute/path/config.py /absolute/path/model.pth /absolute/path/image.png \
  --backbone /absolute/path/dinov3_pretrain.pth \
  --output-dir /absolute/path/predictions
```

The command saves an original-resolution train-ID mask and a colored overlay. IDs 0–18 follow the Cityscapes order: road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle. Evaluation ignores ground-truth ID 255. Official benchmark submissions use their own label-ID format.

## Dataset evaluation

Follow the [data preparation guide](docs/datasets.md) for dataset layout and label conversion. `--data-root` points to the parent directory containing `cityscapes`, `bdd100k`, `mapillary_v1.2`, and `acdc`, as required by the selected configuration. Mapillary uses the prepared 19-class labels at their original resolution.

```bash
python tools/test.py /absolute/path/config.py /absolute/path/model.pth \
  --backbone /absolute/path/dinov3_pretrain.pth \
  --data-root /absolute/path/data \
  --work-dir /absolute/path/evaluation
```

The command saves `metrics.json` with per-domain results. ACDC validation is reported as the arithmetic mean over night, snow, fog, and rain. These are the validation results in Table II; Table VII's ACDC test-server results use a separate protocol.

Use `--cfg-options key=value` for MMEngine configuration overrides. Keep the released preprocessing, class mapping, and sliding-window settings when reproducing the reported scores.

All six released checkpoints use the student branch selected by the original experiments, including the Rein-A model. Inference uses the image, backbone, and released adapter/head weights.

## Training and further releases

The remaining checkpoints and training code will be released progressively. See the [paper coverage table](docs/models.md#paper-coverage) for the experiments included in this batch.

## License

Project-owned code and adapter/head weights use [GPL-3.0](LICENSE.txt). DINOv3-derived files retain the [DINOv3 License](licenses/DINOv3-LICENSE.txt), and third-party components retain their respective licenses. See [third-party notices](docs/third-party-notices.md). Download the base weights separately under Meta's terms.

## Citation

```bibtex
@article{Wei2026ReinPlusPlus,
  author = {Wei, Zhixiang and Ma, Xiaoxiao and Yan, Ruishen and Tu, Tao and Chen, Huaian and Zheng, Jinjin and Jin, Yi and Chen, Enhong},
  title = {Rein++: Efficient Generalization and Adaptation for Semantic Segmentation With Vision Foundation Models},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year = {2026},
  volume = {48},
  number = {8},
  doi = {10.1109/TPAMI.2026.3681631}
}
```
