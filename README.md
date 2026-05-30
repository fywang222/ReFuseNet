# RefSeg

RefSeg 是一个语义分割实验代码库。输入是图像，输出是每个像素的类别 logits；模型内部不做 softmax。训练 loss 由外部模块处理，目前使用 `CrossEntropyLoss(ignore_index=255)`。

## 环境安装

```bash
conda create -n refseg python=3.10 -y
conda activate refseg
pip install -r requirements.txt
export CITYSCAPES_DATASET=/path/to/cityscapes
csCreateTrainIdLabelImgs
```

如果使用 SAM / ReFuseNet，需要额外准备 Meta Segment Anything 的 checkpoint，并在 YAML 配置里写明路径。

## 项目目录

```text
.
├── configs/        # 实验配置。数据路径、模型名、训练参数都在这里改
├── configs/ablation/
│   ├── cityscapes_refusenet_s0.yaml
│   ├── cityscapes_refusenet_s1.yaml
│   ├── cityscapes_refusenet_s2.yaml
│   ├── cityscapes_refusenet_s3.yaml
│   ├── cityscapes_refusenet_s4.yaml
│   └── cityscapes_refusenet_s6.yaml
├── datasets/       # 数据集读取、颜色表、数据增强、image/mask 转 tensor
├── models/         # 模型定义和 build_model 入口
├── losses/         # loss 构建
├── utils/          # metric、checkpoint、可视化、logger、随机种子
├── tools/          # 训练、评估、数据检查、预测可视化入口
├── scripts/        # 常用实验脚本
├── requirements.txt
└── README.md
```

## 数据集安放位置

推荐把数据集放在项目的 `data/` 目录下。也可以放在其他硬盘目录，只要把配置文件里的 `dataset.root` 改成对应路径即可。

推荐结构：

```text
data/
├── CamVid/
│   ├── images/
│   ├── labels/
│   └── list/
│       ├── train.lst
│       ├── val.lst
│       └── test.lst
└── Cityscapes/
    ├── leftImg8bit/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── gtFine/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── list/
        ├── train.lst
        ├── val.lst
        └── test.lst
```

CamVid 配置示例：

```yaml
dataset:
  name: camvid
  root: data/CamVid
  num_classes: 11
  ignore_index: 255
  train_list: list/train.lst
  val_list: list/val.lst
  test_list: list/test.lst
  crop_size: [512, 512]
```

Cityscapes 配置示例：

```yaml
dataset:
  name: cityscapes
  root: data/Cityscapes
  num_classes: 19
  ignore_index: 255
  crop_size: [512, 1024]
```

`train_list`、`val_list`、`test_list` 是可选字段。它们可以写相对 `dataset.root` 的路径，也可以写绝对路径。

list 文件推荐每行写 image 和 mask 两列：

```text
images/0001.png labels/0001_L.png
images/0002.png labels/0002_L.png
```

Cityscapes 也可以写两列：

```text
leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png
```

如果 Cityscapes list 只写 image 路径，代码会根据标准命名自动推断对应的 `gtFine_labelIds` 标签路径。

## Mask 要求

所有 dataset class 返回统一格式：

```python
{
    "image": Tensor[3, H, W],
    "mask": LongTensor[H, W],
    "name": str,
    "orig_size": (H0, W0),
}
```

`mask` 是类别 id，不是 RGB 图。训练 batch 里的 target mask 形状是 `[B, H, W]`，忽略标签统一是 `255`。

CamVid 的 label mask 通常是 RGB 彩色图，代码会根据 `datasets/color_maps.py` 转成 11 类 class id。不属于颜色表的像素会变成 `255`。

Cityscapes 使用 `gtFine_labelIds.png`，不是 `gtFine_color.png`。代码会把 Cityscapes 原始 `labelId` 转成标准 19 类 `trainId`，不属于 19 类的像素会变成 `255`。

## 模型简介

目前支持的模型：

| 模型 | 配置名 | 简介 |
| --- | --- | --- |
| FCN-ResNet50 | `model.name: fcn_resnet50` | torchvision FCN baseline，ResNet50 backbone 使用 ImageNet 预训练权重 |
| SegFormer-B5 | `model.name: segformer_b5` | HuggingFace `nvidia/mit-b5` encoder baseline，用作更强的可比 baseline |
| SAM baseline | `model.name: sam_vitl_decoder` | 只使用 SAM image encoder，接一个简单分割 decoder |
| ReFuseNet | `model.name: ReFuseNet` | 本项目主要 ablation 模型，只使用 SAM image encoder 做普通语义分割 |

ReFuseNet 不使用 SAM prompt encoder，不使用 SAM mask decoder，不使用 point / box / text prompt，也不做条件融合、FiLM 或语言融合。它是纯 image-to-logits 的语义分割模型。

模型 forward 至少返回：

```python
{
    "logits": Tensor[B, num_classes, H, W],
    "coarse_logits": Tensor[B, num_classes, H, W] | None,
    "boundary_logits": Tensor[B, 1, H, W] | None,
}
```

S6 会返回 `boundary_logits`，用于外部可选 boundary loss。开启 `model.debug: true` 时，ReFuseNet 会额外返回 `features` 字段用于调试。训练和评估脚本只依赖 `outputs["logits"]`。

## ReFuseNet Ablation

ReFuseNet 是单一顶层模型类，所有 ablation 都通过配置控制，不需要创建 `ReFuseNetS0`、`ReFuseNetS1` 这类模型变体。

| 设置 | 目的 | 结构 |
| --- | --- | --- |
| S0 | 冻结 SAM encoder 的基础对照 | frozen SAM image encoder + final feature + plain decoder |
| S1 | 观察低学习率 fine-tune 的收益 | low-LR SAM fine-tune + final feature + plain decoder |
| S2 | 控制 pyramid decoder 架构影响 | low-LR SAM fine-tune + pseudo 4-scale features + multi-scale decoder |
| S3 | 观察真实 SAM 中间层特征收益 | low-LR SAM fine-tune + true 4-level SAM features + multi-scale decoder |
| S4 | 观察 refinement 收益 | S3 + GRU-style iterative refinement |
| S6 | 观察 DA3 DualDPT 解码和边界监督收益 | low-LR SAM fine-tune + DA3-style DualDPT decoder + PIDNet-style boundary head |

高层配置示例：

```yaml
model:
  name: ReFuseNet
  setting: S0
```

显式配置示例：

```yaml
model:
  name: ReFuseNet
  setting: S3
  sam:
    model_type: vit_b
    checkpoint: /path/to/sam_vit_b.pth
    train_mode: low_lr_ft
    intermediate_blocks: [3, 6, 9, 12]
  decoder:
    num_classes: 19
    dim: 128
    feature_mode: multi_level
    fusion_mode: multiscale
    pseudo_scales: [4, 8, 16, 32]
  refine:
    enabled: false
    type: gru
    iters: 3
```

`setting` 会给出默认 preset。如果同时写了 `sam`、`decoder`、`refine` 里的低层字段，显式字段会覆盖 preset 默认值。

S4 开启 refinement 后，`logits` 是 refinement 后的结果，`coarse_logits` 是 refinement 前的粗分割结果，可用于可选辅助 loss。

S6 使用四层 SAM 中间特征，按 DA3 `DualDPT` 思路做四尺度投影、重采样和双分支 DPT 融合：主分支输出语义 `logits`，独立辅助分支输出一通道 `boundary_logits`。边界标签生成和 boundary loss 不在模型里做，建议在 trainer/loss 模块中按 PIDNet 类似方式额外监督。

## 预训练权重

不同模型的预训练来源不同：

| 模型 | 预训练来源 | 是否必须手动下载 |
| --- | --- | --- |
| FCN-ResNet50 | torchvision ImageNet ResNet50 backbone 权重 | 否。第一次运行时 torchvision 会自动下载 |
| SegFormer-B5 | HuggingFace `nvidia/mit-b5` | 否。第一次运行时 transformers 会自动下载 |
| SAM baseline / ReFuseNet | Meta Segment Anything 官方 checkpoint | 是。需要手动下载并在 YAML 里设置路径 |

ReFuseNet 使用 SAM ViT-B 时：

```yaml
model:
  name: ReFuseNet
  setting: S3
  sam:
    model_type: vit_b
    checkpoint: /absolute/path/to/sam_vit_b.pth
```

SAM baseline 使用 ViT-L 时：

```yaml
model:
  name: sam_vitl_decoder
  num_classes: 19
  checkpoint: /absolute/path/to/sam_vit_l.pth
  freeze_encoder: true
```

如果 SAM checkpoint 没填，相关模型会直接报错；这是故意的，因为 SAM 权重较大，不适合自动下载。

## 训练

训练 ReFuseNet S0：

```bash
python tools/train.py --config configs/ablation/cityscapes_refusenet_s0.yaml
```

训练 ReFuseNet S3：

```bash
python tools/train.py --config configs/ablation/cityscapes_refusenet_s3.yaml
```

训练 ReFuseNet S4：

```bash
python tools/train.py --config configs/ablation/cityscapes_refusenet_s4.yaml
```

训练 ReFuseNet S6：

```bash
python tools/train.py --config configs/ablation/cityscapes_refusenet_s6.yaml
```

训练 CamVid FCN baseline：

```bash
python tools/train.py --config configs/camvid_fcn_resnet50.yaml
```

训练所有 CamVid baselines：

```bash
bash scripts/train_camvid_baselines.sh
```

训练所有 Cityscapes baselines：

```bash
bash scripts/train_cityscapes_baselines.sh
```

训练过程会保存：

```text
outputs/<experiment_name>/
├── last.pth
├── best.pth
├── checkpoints/
│   └── epoch_0010.pth
└── run.log
```

`last.pth` 是最后一个 epoch。`best.pth` 是 validation mIoU 最好的 checkpoint。所有训练 checkpoint 都包含模型、optimizer 和 AMP scaler 状态。

按固定间隔保存 epoch checkpoint：

```bash
python tools/train.py \
  --config configs/camvid_fcn_resnet50.yaml \
  --save-every 10
```

只保存指定 epoch：

```bash
python tools/train.py \
  --config configs/camvid_fcn_resnet50.yaml \
  --save-epochs 10,20,40
```

## 恢复训练和加载预训练

从上次中断处继续训练：

```bash
python tools/train.py \
  --config configs/camvid_fcn_resnet50.yaml \
  --resume outputs/camvid_fcn_resnet50/last.pth
```

也可以按 epoch 恢复，下面会加载 `outputs/camvid_fcn_resnet50/checkpoints/epoch_0010.pth`：

```bash
python tools/train.py \
  --config configs/camvid_fcn_resnet50.yaml \
  --resume 10
```

覆盖配置里的总训练 epoch 数：

```bash
python tools/train.py \
  --config configs/camvid_fcn_resnet50.yaml \
  --epochs 120
```

从另一个 checkpoint 做 shape-matched partial loading：

```bash
python tools/train.py \
  --config configs/camvid_fcn_resnet50.yaml \
  --pretrained outputs/cityscapes_fcn_resnet50/best.pth
```

`--pretrained` 会只加载名字相同、shape 也相同的 tensor。分类头通常因为类别数不同而被跳过，这正是 Cityscapes 到 CamVid fine-tuning 需要的行为。

## 评估

```bash
python tools/eval.py \
  --config configs/camvid_fcn_resnet50.yaml \
  --ckpt outputs/camvid_fcn_resnet50/best.pth \
  --save-pred
```

评估会打印 mIoU、pixel accuracy、mean accuracy、per-class IoU。CamVid 还会额外统计 rare mIoU：`Pole`、`SignSymbol`、`Pedestrian`、`Bicyclist`。

如果加 `--save-pred`，预测图会保存到：

```text
outputs/<experiment_name>/predictions/<split>/
```

## 单独保存预测可视化

```bash
python tools/visualize_predictions.py \
  --config configs/camvid_fcn_resnet50.yaml \
  --ckpt outputs/camvid_fcn_resnet50/best.pth \
  --num-samples 8
```

每个样本会保存：

- `<name>_image.png`
- `<name>_gt.png`
- `<name>_pred.png`
- `<name>_error.png`
- `<name>_overlay.png`

## 参考

- HRNet Semantic Segmentation: https://github.com/HRNet/HRNet-Semantic-Segmentation
- torchvision pretrained weights: https://docs.pytorch.org/vision/stable/models.html
- SegFormer-B5: https://huggingface.co/nvidia/mit-b5
- Segment Anything: https://github.com/facebookresearch/segment-anything
