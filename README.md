# RefSeg

RefSeg 是一个干净、轻量的 PyTorch semantic segmentation 项目骨架。第一阶段目标不是实现最终方法，而是先让一个没有 AI 项目经验的人也能看懂并跑通完整流程：

1. 准备数据集。
2. 检查 image/mask 是否匹配。
3. 训练一个 baseline。
4. 评估 checkpoint。
5. 保存预测图、GT 图和 error map。

当前支持：

- CamVid 11 类语义分割。
- Cityscapes 19 类语义分割。
- FCN-ResNet50 baseline。
- SegFormer-B1 baseline。
- SAM ViT-L encoder + simple decoder baseline。

本项目不使用 MMLab，也不使用 PaddleSeg。

## 安装环境

推荐使用 conda：

```bash
conda env create -f environment.yml
conda activate refseg
```

`environment.yml` 默认使用 CPU 版 PyTorch，适合先检查代码和数据。如果要用 GPU 训练，请根据机器 CUDA 版本安装对应的 PyTorch / torchvision。可以参考 PyTorch 官网安装命令。

项目也保留了 `requirements.txt`，用于查看最小 Python 包依赖。

## 预训练权重

不同模型的预训练来源不同，不要混在一起理解。

| 模型 | 代码配置 | 预训练来源 | 是否必须手动下载 |
| --- | --- | --- | --- |
| FCN-ResNet50 | `model.name: fcn_resnet50` | torchvision 的 ImageNet ResNet50 backbone 权重，代码中使用 `ResNet50_Weights.IMAGENET1K_V2` | 否。第一次运行时 torchvision 会自动下载到本机缓存 |
| SegFormer-B1 | `model.name: segformer_b1` | HuggingFace `nvidia/mit-b1`，这是 ImageNet-1k 预训练的 MiT-B1 encoder | 否。第一次运行时 transformers 会自动下载到 HuggingFace 缓存 |
| SAM ViT-L | `model.name: sam_vitl_decoder` | Meta Segment Anything 官方 ViT-L checkpoint | 是。需要手动下载并在 YAML 里设置 `model.checkpoint` |

常用链接：

- torchvision pretrained weights 文档：https://docs.pytorch.org/vision/stable/models.html
- SegFormer-B1 HuggingFace 页面：https://huggingface.co/nvidia/mit-b1
- Segment Anything 官方仓库：https://github.com/facebookresearch/segment-anything
- SAM ViT-L checkpoint：https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

SAM 配置示例：

```yaml
model:
  name: sam_vitl_decoder
  num_classes: 11
  checkpoint: /absolute/path/to/sam_vit_l_0b3195.pth
  freeze_encoder: true
```

如果 `checkpoint` 没填，SAM baseline 会直接报清楚的错误；这是故意的，因为 SAM 权重较大，不适合偷偷自动下载。

## 项目目录

```text
.
├── configs/        # 实验配置。数据路径、模型名、训练轮数、batch size 都在这里改
├── datasets/       # 数据集读取、颜色表、数据增强和 image/mask 转 tensor
├── models/         # 模型定义和 build_model 入口
├── losses/         # loss 构建，目前是 CrossEntropyLoss(ignore_index=255)
├── utils/          # metric、checkpoint、可视化、logger、随机种子
├── tools/          # 可直接运行的命令行入口：训练、评估、数据检查、预测可视化
├── scripts/        # 常用实验脚本
├── environment.yml # conda 环境
├── requirements.txt
└── README.md
```

各目录作用：

- `configs/`：一个 YAML 文件对应一个实验。新实验优先复制已有 YAML 再改字段。
- `datasets/`：只负责把硬盘上的图片和标签读成统一格式，不写训练逻辑。
- `models/`：只负责模型 forward。所有模型 forward 都返回 `{"logits": Tensor[B,C,H,W], "aux": dict}`。
- `losses/`：训练时怎么从 logits 和 mask 计算 loss。
- `utils/metrics.py`：统计 mIoU、pixel accuracy、mean accuracy。会忽略 label 为 `255` 的像素。
- `utils/checkpoint.py`：保存和加载 checkpoint，支持 shape-matched partial loading，方便 Cityscapes 到 CamVid fine-tuning。
- `utils/visualization.py`：保存输入图、GT mask、预测 mask、错误图和 overlay。
- `tools/train.py`：训练入口。
- `tools/eval.py`：评估入口。
- `tools/check_dataset.py`：训练前检查数据集入口。
- `tools/visualize_predictions.py`：加载 checkpoint 保存可视化结果。

## 统一数据格式

所有 dataset class 都返回同一种 sample：

```python
{
    "image": Tensor[3, H, W],
    "mask": LongTensor[H, W],
    "name": str,
    "orig_size": (H0, W0),
}
```

这里的 `mask` 是 class id，不是 RGB 图。被忽略的像素统一为 `255`。

所有 model forward 都返回同一种 output：

```python
{
    "logits": Tensor[B, C, H, W],
    "aux": dict,
}
```

如果某个模型额外输出 decoder feature，可以多返回：

```python
{
    "logits": Tensor[B, C, H, W],
    "aux": dict,
    "features": Tensor[B, D, h, w],
}
```

训练和评估脚本只依赖 `outputs["logits"]`，所以不会绑定某个特殊模型。

## 数据集配置方式

这个项目的数据配置思路参考 HRNet Semantic Segmentation：配置里明确写 `dataset.root`，必要时用 list 文件控制 train / val / test 样本集合。HRNet README 中的数据准备方式也是把数据放在 `$SEG_ROOT/data` 下，并用 `list/cityscapes/train.lst` 这类文件管理 split。

本项目支持两种方式：

1. 使用标准数据目录，让代码自动扫描。
2. 使用显式 list 文件，推荐给新手和多人协作项目，因为更可控。

### YAML 字段说明

```yaml
dataset:
  name: camvid              # camvid 或 cityscapes
  root: /path/to/CamVid     # 数据集根目录，不要在源码里写死机器路径
  num_classes: 11           # CamVid 是 11，Cityscapes 是 19
  ignore_index: 255         # 被忽略像素的 label id
  train_split: train
  val_split: val
  test_split: test
  train_list: null          # 可选。显式 train list 文件
  val_list: null            # 可选。显式 val list 文件
  test_list: null           # 可选。显式 test list 文件
  crop_size: [512, 512]
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
```

`root` 是最重要的字段。换机器时通常只需要改 `root` 和 SAM checkpoint 路径。

### 推荐的数据目录

可以把数据放在项目的 `data/` 下面，也可以放到任意大硬盘目录。推荐布局如下：

```text
data/
├── camvid/
│   ├── images/
│   ├── labels/
│   └── list/
│       ├── train.lst
│       ├── val.lst
│       └── test.lst
└── cityscapes/
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

如果数据放在这里，对应配置可以写：

```yaml
dataset:
  name: camvid
  root: data/camvid
  train_list: list/train.lst
  val_list: list/val.lst
  test_list: list/test.lst
```

路径规则：`train_list` 是相对 `dataset.root` 的路径。如果写绝对路径也可以。

### list 文件格式

每行一个样本，推荐写 image 和 mask 两列：

```text
images/0001.png labels/0001_L.png
images/0002.png labels/0002_L.png
```

Cityscapes 也可以写两列：

```text
leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png
```

如果 Cityscapes list 只写 image 路径，代码会根据标准命名自动推断 `gtFine_labelIds` 标签路径。

### CamVid mask 要求

CamVid 的 label mask 通常是 RGB 彩色图。代码会把 RGB 颜色转换成 class id：

```python
CAMVID_CLASSES = [
    "Sky", "Building", "Pole", "Road", "Pavement", "Tree",
    "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist",
]
```

颜色表在 `datasets/color_maps.py`。不属于这 11 类颜色的像素会被设置成 `255`，训练和评估时会被忽略。

### Cityscapes mask 要求

Cityscapes 使用 `gtFine_labelIds.png`，不是 `gtFine_color.png`。代码会把 Cityscapes 原始 `labelId` 转成标准 19 类 `trainId`：

```python
CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]
```

不属于 19 类的 label 会变成 `255`。

## 训练前必须检查数据

新手最常见的问题不是模型写错，而是 image/mask 没对上、mask 颜色不对、路径不对、ignore label 不对。训练前先跑：

```bash
python tools/check_dataset.py --config configs/camvid_fcn_resnet50.yaml
```

这个脚本会输出：

- dataset size。
- sample image shape。
- sample mask shape。
- mask unique values。
- class histogram。
- 若干可视化样例。

输出图片会保存到：

```text
outputs/<experiment_name>/dataset_check/
```

如果看到 mask 几乎全是黑色、unique values 只有 `255`，说明 label 颜色映射或 label 路径大概率错了。

## Debug Overfit

第一次跑项目时，不要直接训练完整数据集。先用 2 张图 overfit：

```bash
python tools/train.py --config configs/debug_overfit_camvid.yaml
```

这个配置里有：

```yaml
dataset:
  overfit_num_samples: 2

train:
  epochs: 50
  batch_size: 2
```

如果 2 张图都无法明显 overfit，优先检查：

- mask 是否读成了 class id。
- `num_classes` 是否正确。
- `ignore_index` 是否正确。
- model 输出尺寸是否和 mask 一致。
- loss 是否下降。

## 训练

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
└── run.log
```

`last.pth` 是最后一个 epoch。`best.pth` 是 validation mIoU 最好的 checkpoint。

## 恢复训练和加载预训练

从上次中断处继续训练：

```bash
python tools/train.py \
  --config configs/camvid_fcn_resnet50.yaml \
  --resume outputs/camvid_fcn_resnet50/last.pth
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

评估会打印：

- mIoU。
- pixel accuracy。
- mean accuracy。
- per-class IoU。
- CamVid rare mIoU：`Pole`、`SignSymbol`、`Pedestrian`、`Bicyclist`。

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

## 常见问题

### 1. `python: command not found`

有些机器只提供 `python3`。激活 conda 环境后一般会有 `python`。先确认：

```bash
conda activate refseg
which python
python --version
```

### 2. 找不到 `torch`

说明当前 shell 没进正确的 conda 环境，或者环境没有创建成功：

```bash
conda activate refseg
python -c "import torch; print(torch.__version__)"
```

### 3. GPU 不能用

先检查：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

如果输出 `False`，常见原因是安装了 CPU 版 PyTorch，或者 CUDA driver / PyTorch CUDA 版本不匹配。

### 4. 第一次运行很慢

FCN 和 SegFormer 第一次会下载预训练权重。下载缓存通常在：

- torchvision / torch hub：`~/.cache/torch`
- HuggingFace：`~/.cache/huggingface`

服务器不能联网时，需要提前下载好权重，或者把模型配置里的 `pretrained: false` 改掉做代码检查。

### 5. SAM 报 checkpoint not found

SAM 权重不会自动下载。下载：

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```

然后在 YAML 中设置：

```yaml
model:
  checkpoint: /path/to/sam_vit_l_0b3195.pth
```

### 6. mask unique values 很奇怪

正常情况下，CamVid mask 的有效 id 应该在 `0..10`，Cityscapes mask 的有效 id 应该在 `0..18`，ignore 是 `255`。

如果出现很多大于类别数的值，说明你可能读到了 RGB mask 的单通道、color mask、或者错误的 label 文件。

### 7. loss 是 NaN

优先检查：

- 学习率是否过大。
- mask 是否全是 `255`。
- image 是否正常归一化。
- AMP 是否导致不稳定。可以先把 `train.amp: false`。

### 8. mIoU 一直是 0

优先检查：

- `num_classes` 是否和数据集一致。
- 模型输出通道数是否等于 `num_classes`。
- mask 是否正确转换成 class id。
- validation list 是否为空或路径错误。

### 9. DataLoader 卡住

把 `num_workers` 先改成 0：

```yaml
train:
  num_workers: 0
```

这对定位路径、PIL 读图、权限问题更直接。

### 10. 显存不够

先降低：

```yaml
train:
  batch_size: 1
  crop_size: [384, 384]
```

SAM ViT-L 显存占用明显更高。第一阶段建议先用 FCN-ResNet50 或 SegFormer-B1 跑通流程。

## 给新同学的推荐流程

按这个顺序走，不要跳步：

1. 创建 conda 环境。
2. 修改 config 里的 `dataset.root`。
3. 运行 `tools/check_dataset.py`。
4. 查看 `outputs/.../dataset_check/` 里的可视化。
5. 运行 `configs/debug_overfit_camvid.yaml`。
6. 确认 loss 能下降。
7. 训练正式 baseline。
8. 用 `tools/eval.py` 评估 `best.pth`。
9. 用 `tools/visualize_predictions.py` 看预测质量。

## 参考

- HRNet Semantic Segmentation：https://github.com/HRNet/HRNet-Semantic-Segmentation
- torchvision pretrained weights：https://docs.pytorch.org/vision/stable/models.html
- SegFormer-B1：https://huggingface.co/nvidia/mit-b1
- Segment Anything：https://github.com/facebookresearch/segment-anything
