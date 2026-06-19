# ReFuseNet

Semantic segmentation experiments on CamVid and Cityscapes. Training uses raw logits with `CrossEntropyLoss(ignore_index=255)`.

## Setup

```bash
conda create -n refusenet python=3.10 -y
conda activate refusenet
pip install -r requirements.txt
```

ReFuseNet needs the SAM ViT-B checkpoint here:

```text
checkpoints/sam_vit_b_01ec64.pth
```

W&B is enabled by default:

```bash
export WANDB_API_KEY=<wandb_api_key>
export WANDB_PROJECT=refusenet
```

Disable per run with `--wandb off`.

## Data

Expected layout:

```text
data/
├── CamVid/
│   ├── train/images/
│   ├── train/labels/
│   ├── test/images/
│   └── test/labels/
└── Cityscapes/
    ├── leftImg8bit/
    │   ├── train/<city>/
    │   └── val/<city>/
    └── gtFine/
        ├── train/<city>/
        └── val/<city>/
```

CamVid masks may be grayscale ids or RGB masks. Valid classes are `0..10`; void is `255`.

Cityscapes configs use `dataset.label_format: trainIds`, so they read `*_gtFine_labelTrainIds.png` directly. Generate those masks with the official scripts:

```bash
pip install cityscapesscripts
CITYSCAPES_DATASET=data/Cityscapes python -m cityscapesscripts.preparation.createTrainIdLabelImgs
```

The Cityscapes loader only scans `leftImg8bit/<split>` and never falls back to `gtFine` images.

## Models

| Setting | Structure |
| --- | --- |
| FCN | torchvision FCN-ResNet50 |
| SegFormer-B5 | HuggingFace `nvidia/mit-b5` |
| S0 | frozen SAM + final feature + legacy decoder |
| S1 | SAM low-LR fine-tune + final feature |
| S2 | SAM low-LR fine-tune + pseudo pyramid `256/128/64/32` |
| S3 | SAM intermediate layers + multiscale fusion |
| S4 | SAM intermediate layers + DPT fusion |
| S5 | S4 + GRU refinement at `256x256` head logits |
| S6 | S2 + boundary head |
| S7 | S2 + GRU refinement |
| C2 | SAM final feature + lightweight CNN `256/128` scales + SAM-projected `64/32` scales + FPN fusion |
| C4 | C2 + GRU refinement; GRU context uses only SAM final feature projection |
| T1 | SAM final feature + lightweight Segmenter-style transformer decoder |

For S1-S7 and C2/C4, the classifier input is always:

```text
semantic_features: [B, head_channels=64, 256, 256]
```

S5/S7 refine head-resolution logits, not full-resolution logits. S6 adds `lambda_boundary * BCEWithLogits(boundary_logits, boundary_target)`.
C4 follows the same head-resolution GRU path, but uses a SAM-final-only `gru_context` instead of the fused decoder feature as refinement context.

T1 replaces the whole semantic decoder with a compact Segmenter-style mask transformer:

```text
SAM final feature: [B, 256, 64, 64]
patch tokens + class tokens -> 4 transformer blocks -> logits [B, num_classes, 64, 64]
```

T1 does not use the shared classifier, GRU, CNN pyramid, DPT fusion, or boundary head.

## Config Defaults

| Dataset | Epochs | Eval Every | Save Every | Eval |
| --- | ---: | ---: | ---: | --- |
| CamVid | 200 | 10 | 50 | whole image |
| Cityscapes | 50 | 5 | 50 | sliding, crop `1024`, stride `768`, eval batch `1` |

Batch and accumulation:

| Model | Batch | Grad Accum |
| --- | ---: | ---: |
| FCN | 8 | 1 |
| SegFormer-B5 | 4 | 2 |
| ReFuseNet S0-S7/C2/C4/T1 | 2 | 4 |

All configs use AdamW with linear warmup and poly LR decay. Scheduler steps on optimizer steps, so it respects gradient accumulation.

## Training

Single run:

```bash
python tools/train.py --config configs/camvid/camvid_refusenet_s2.yaml --device cuda
```

CamVid full sweep, default GPUs `1 2 3`, with dynamic refill when a job finishes:

```bash
bash scripts/train_camvid_baselines.sh
```

Cityscapes full sweep, default GPUs `4 5 6 7`:

```bash
bash scripts/train_cityscapes_baselines.sh
```

Cityscapes-to-CamVid 20 epoch fine-tune on GPU 0:

```bash
bash scripts/train_camvid_from_cityscapes_20epoch_gpu0.sh
```

C2/C4/T1 supplement runs, default GPUs `2 3 4 5 6 7` for three CamVid and three Cityscapes jobs:

```bash
bash scripts/train_refusenet_c2_c4_t1_supplement.sh
```

S5 debug, eval every epoch:

```bash
python tools/train.py --config configs/cityscapes/cityscapes_refusenet_s5.yaml --device cuda --s5-debug --eval-every 1
```

Outputs:

```text
outputs/<experiment_name>/
├── last.pth
├── checkpoints/epoch_0050.pth
└── run.log
```

Resume:

```bash
python tools/train.py --config configs/camvid/camvid_refusenet_s2.yaml --resume outputs/camvid_refusenet_s2/last.pth --device cuda
python tools/train.py --config configs/camvid/camvid_refusenet_s2.yaml --resume 50 --device cuda
```

## Evaluation

```bash
python tools/eval.py \
  --config configs/camvid/camvid_refusenet_s2.yaml \
  --ckpt outputs/camvid_refusenet_s2/last.pth \
  --device cuda \
  --save-pred
```

Metric logic:

- accumulate one full-validation `num_classes x num_classes` confusion matrix
- ignore `target == 255`
- non-ignore targets and predictions outside `[0, num_classes)` raise errors
- IoU is `diag / (row_sum + col_sum - diag)`
- `union == 0` classes are `NaN`
- mIoU is `np.nanmean(per_class_iou)`

## References

- torchvision pretrained weights: https://docs.pytorch.org/vision/stable/models.html
- SegFormer-B5: https://huggingface.co/nvidia/mit-b5
- Segment Anything: https://github.com/facebookresearch/segment-anything
