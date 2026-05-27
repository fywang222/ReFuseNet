#!/usr/bin/env bash
set -e

python tools/train.py \
  --config configs/camvid_fcn_resnet50.yaml \
  --pretrained outputs/cityscapes_fcn_resnet50/best.pth

