#!/usr/bin/env bash
set -e

python tools/train.py --config configs/camvid_fcn_resnet50.yaml
python tools/train.py --config configs/camvid_segformer_b1.yaml
python tools/train.py --config configs/camvid_sam_vitl.yaml

