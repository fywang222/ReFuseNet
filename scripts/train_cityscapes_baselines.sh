#!/usr/bin/env bash
set -e

python tools/train.py --config configs/cityscapes_fcn_resnet50.yaml
python tools/train.py --config configs/cityscapes_segformer_b5.yaml
python tools/train.py --config configs/cityscapes_sam_vitl.yaml
