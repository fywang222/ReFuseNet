# ReFuseNet

Semantic segmentation experiments for CamVid and Cityscapes. Models return raw per-pixel logits; training uses `CrossEntropyLoss(ignore_index=255)` directly in `tools/train.py`.

## Environment

```bash
conda create -n refusenet python=3.10 -y
conda activate refusenet
pip install -r requirements.txt
```

For ReFuseNet, download the official Meta Segment Anything SAM ViT-B checkpoint and provide its absolute path through `SAM_VIT_B_CHECKPOINT`.

```bash
export SAM_VIT_B_CHECKPOINT=/absolute/path/to/sam_vit_b.pth
```

W&B is enabled by default. Install `wandb` from `requirements.txt` and set:

```bash
export WANDB_API_KEY=<wandb_api_key>
export WANDB_PROJECT=refusenet
```

Use `--wandb off` if you need to disable W&B for one run.

## Repository Layout

```text
.
├── configs/        # experiment configs
├── datasets/       # dataset indexing, label mapping, transforms
├── models/         # FCN, SegFormer-B5, ReFuseNet
├── scripts/        # batch training scripts
├── tools/          # train/eval/check/visualization entry points
├── utils/          # checkpoints, metrics, logging, visualization
├── requirements.txt
└── README.md
```

## Data Layout

Current expected local layout:

```text
data/
├── CamVid/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
└── Cityscapes/
    ├── leftImg8bit/
    │   ├── train/<city>/
    │   └── val/<city>/
    └── gtFine/
        ├── train/<city>/
        └── val/<city>/
```

CamVid labels may be grayscale class ids or RGB masks. Valid classes are `0..10`; invalid/void ids are mapped to `255`.

Cityscapes uses original `leftImg8bit` images as inputs and `*_gtFine_labelIds.png` masks as targets, mapping original labelIds to 19 trainIds. `gtFine` color or label files are not used as image-input fallbacks.

## Transforms And Evaluation

CamVid:

```text
train:    Resize(360,480) -> RandomHorizontalFlip(0.5) -> ToTensor -> Normalize
val/test: Resize(360,480) -> ToTensor -> Normalize
eval:     whole-image inference
```

Cityscapes:

```text
train:    RandomResize(scale=(2048,1024), ratio_range=(0.5,2.0), keep_ratio=True)
          -> RandomCrop(1024,1024)
          -> RandomHorizontalFlip(0.5)
          -> PhotoMetricDistortion
          -> ToTensor -> Normalize
val/test: whole image input, no resize
eval:     sliding-window inference, crop=1024, stride=768
```

## Models

| Model | Config key | Notes |
| --- | --- | --- |
| FCN-ResNet50 | `model.name: fcn_resnet50` | Requires torchvision pretrained ResNet50 backbone weights. Failure to load weights raises an error. |
| SegFormer-B5 | `model.name: segformer_b5` | Requires HuggingFace `nvidia/mit-b5` pretrained weights. Failure to load weights raises an error. |
| ReFuseNet | `model.name: ReFuseNet` | Uses SAM ViT-B image encoder. `model.sam.checkpoint` is mandatory. |

No model silently falls back to random initialization for pretrained backbones.

## ReFuseNet Roadmap

ReFuseNet remains a single configurable top-level class. Do not create `ReFuseNetS4`, `ReFuseNetS5`, etc.

| Setting | Structure |
| --- | --- |
| S0 | frozen SAM encoder + final feature + plain decoder |
| S1 | low-LR SAM fine-tune + final feature + plain decoder |
| S2 | low-LR SAM fine-tune + pseudo 4-scale features + naive multiscale decoder |
| S3 | low-LR SAM fine-tune + true 4-level SAM features + naive multiscale decoder |
| S4 | low-LR SAM fine-tune + true 4-level SAM features + DPT-style semantic decoder |
| S5 | S4 + GRU-style iterative refinement |
| S6 | DA3-style Dual-DPT dual-branch boundary decoder, kept in code/config but not run by default scripts |

Decoder fields:

```yaml
decoder:
  dim: 128
  head_channels: 64
  head_resolution: [256, 256]
  feature_mode: final | pseudo_pyramid | multi_level | dualdpt
  fusion_mode: single | multiscale | dpt | dualdpt
  reassembly_channels: [96, 192, 384, 768]
  dpt_head_scale: p1
```

S1-S6 all produce `semantic_features` with shape `[B, head_channels, 256, 256]` before the shared classifier. S0 keeps the legacy frozen-SAM baseline head. S5 refines classifier head logits at `[B, C, 256, 256]`, not full-resolution logits.

Forward outputs:

- All settings return `logits`.
- Only `refine.enabled: true` returns `coarse_logits`, `aux_logits`, `coarse_logits_head`, and `aux_logits_head`.
- Only S6 / `feature_mode: dualdpt` returns `boundary_logits`.
- `debug: true` adds feature tensors for inspection.

SAM preprocessing inside ReFuseNet:

- resize the input so the longest side becomes 1024
- keep aspect ratio
- pad to `1024 x 1024`
- run the SAM encoder
- remove the padding from the logits
- upsample logits back to the input `H x W`

Training loss:

- `logits` always use `CrossEntropyLoss(ignore_index=255)`
- S5 adds `lambda_aux * CE(coarse_logits, mask)`
- S6 adds `lambda_boundary * BCEWithLogits(boundary_logits, boundary_target)`
- `--s5-debug` changes only S5 runs: it adds per-refinement auxiliary losses controlled by `lambda_refine_aux`, leaves `lambda_coarse_aux` at `0.0` by default, and prints refinement diagnostics.
- `tools/train.py` calls `model.get_param_groups(cfg)` when available, so `lr_sam` and `lr_decoder` are respected
- The S5/S6 weights live in `train.lambda_aux`, `train.lambda_refine_aux`, `train.lambda_coarse_aux`, and `train.lambda_boundary`

## Main Training Scripts

CamVid runs FCN, SegFormer-B5, and ReFuseNet S0-S5 for 200 epochs by default:

```bash
export WANDB_API_KEY=<wandb_api_key>
export SAM_VIT_B_CHECKPOINT=/absolute/path/to/sam_vit_b.pth
export PYTHON=python
bash scripts/train_camvid_baselines.sh
```

Cityscapes runs FCN, SegFormer-B5, and ReFuseNet S0-S5 using each config's `train.epochs` value:

```bash
export WANDB_API_KEY=<wandb_api_key>
export SAM_VIT_B_CHECKPOINT=/absolute/path/to/sam_vit_b.pth
export PYTHON=python
bash scripts/train_cityscapes_baselines.sh
```

CamVid script defaults:

```bash
EPOCHS=200
DEVICE=cuda
GPUS="0 1 2 3"
```

Cityscapes script uses each config's `train.epochs`, defaults to `DEVICE=cuda`, and runs on `GPUS="4 5 6 7"` unless overridden.

All configs save epoch checkpoints every 50 epochs. Training evaluates according to `train.eval_every` unless overridden with `--eval-every`: CamVid uses 10, Cityscapes uses 5. Cityscapes uses `val` by default; CamVid uses `test` by default.

## Single Run

```bash
python tools/train.py \
  --config configs/camvid/camvid_fcn_resnet50.yaml \
  --device cuda \
  --epochs 200 \
  --save-every 50
```

S5 debug, evaluating and printing debug metrics every epoch:

```bash
python tools/train.py --config configs/cityscapes/cityscapes_refusenet_s5.yaml --device cuda --s5-debug --eval-every 1
```

Checkpoint outputs:

```text
outputs/<experiment_name>/
├── last.pth
├── checkpoints/
│   └── epoch_0050.pth
└── run.log
```

Resume examples:

```bash
python tools/train.py --config configs/camvid/camvid_fcn_resnet50.yaml --resume outputs/camvid_fcn_resnet50/last.pth
python tools/train.py --config configs/camvid/camvid_fcn_resnet50.yaml --resume 50
```

## Evaluation

```bash
python tools/eval.py \
  --config configs/camvid/camvid_fcn_resnet50.yaml \
  --ckpt outputs/camvid_fcn_resnet50/last.pth \
  --save-pred
```

Metrics include mIoU, pixel accuracy, mean accuracy, and per-class IoU. CamVid also reports rare mIoU for `Pole`, `SignSymbol`, `Pedestrian`, and `Bicyclist`.

## Visualization

```bash
python tools/visualize_predictions.py \
  --config configs/camvid/camvid_fcn_resnet50.yaml \
  --ckpt outputs/camvid_fcn_resnet50/last.pth \
  --num-samples 8
```

## References

- torchvision pretrained weights: https://docs.pytorch.org/vision/stable/models.html
- SegFormer-B5: https://huggingface.co/nvidia/mit-b5
- Segment Anything: https://github.com/facebookresearch/segment-anything
