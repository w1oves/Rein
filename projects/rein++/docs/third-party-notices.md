# Third-party notices

- DINOv3: Meta Platforms, Inc. and affiliates. `reinpp/models/backbones/dino_v3.py` and `dinov3_layers/` are adapted from the original experiment implementation based on DINOv3. Their original DINOv3 License notices apply; see `licenses/DINOv3-LICENSE.txt`. Local modifications add segmentation feature outputs and registry integration; upstream numerical behavior is validated separately.
- OpenMMLab MMSegmentation, MMDetection, MMCV, and MMEngine: Apache-2.0; see [license text](../licenses/Apache-2.0.txt). The Mask2Former forward implementation and multi-domain metrics derive from OpenMMLab-based experiment code. Existing copyright notices are retained. Dependencies are installed from upstream distributions.
- The original Rein repository uses GPL-3.0. Rein adapter code and this release's integration utilities follow that license.
- Mapillary class grouping in `reinpp/labels.py` follows the experiment converter derived from [openseg-group/openseg.pytorch](https://github.com/openseg-group/openseg.pytorch), Copyright (c) 2020 openseg-group (Yuhui Yuan, Jingyi Xie, Jianyuan Guo, Lang Huang), MIT; see [license text](../licenses/OpenSeg-MIT.txt). The release provides a validation-label-only command and preserves original annotation files.

The project-level GPL license does not replace third-party licenses. Pretrained backbone weights are not bundled.
