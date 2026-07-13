# MedSegDiff × MosMed：基于扩散模型的 COVID-19 肺部病灶分割

> 将 [MedSegDiff-V2](https://github.com/WuJunde/MedSegDiff)（AAAI 2024）扩散分割框架适配到 [MosMedData](https://github.com/ncov/MosMedData) 胸部 CT 数据集，完成从数据接入、训练、推理到评估的端到端流程，并对原始框架做了 5 项关键修复以适配 CT 数据与二值病灶分割任务。

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

| 指标 | 值 |
|:---:|:---:|
| 训练步数 | 73,000 |
| 平均 Dice | **0.290** |
| 最佳阈值 Dice | 0.297 |
| 平均 IoU | 0.184 |
| 最佳单切片 Dice | 0.738 |

---

## 1. 项目成果

| 训练步数 | 训练损失 | 平均 Dice | 最佳阈值 Dice | 平均 IoU |
|:---:|:---:|:---:|:---:|:---:|
| **73,000** | **0.236** | **0.290** | **0.297** | **0.184** |

- **数据集**：MosMed Chest CT（50 例带掩码 CT-1 研究，study_0255–0304）
- **任务**：二值病灶分割（前景像素占比约 0.5%，类别极度不均衡）
- **最佳单切片**：study_0267 slice15，Dice=0.74

---

## 2. 环境配置

### 2.1 硬件
- GPU：NVIDIA GeForce RTX 2080 Ti（11 GB 显存）
- 配置：`batch_size=2`、`image_size=256`、`num_channels=128`，训练显存约 4-5 GB

### 2.2 软件
```
Python 3.10
PyTorch 2.5.1+cu121
CUDA 12.1
```

安装依赖（项目根目录）：
```cmd
pip install -r requirement.txt
pip install torchsummary visdom
```

> 注：训练脚本 `scripts/segmentation_train.py` 会启动 visdom 服务（端口 8850）。如不需要可视化，可注释掉第 22 行的 `viz = Visdom(...)`，不影响训练本身。

---

## 3. 数据准备

### 3.1 下载 MosMed 数据
从 [MosMedData GitHub](https://github.com/ncov/MosMedData) 下载，解压后目录结构应为：
```
MosMedData-Chest-CT-Scans-with-COVID-19-Related-Findings-main/
├── dataset_registry.csv          # study_file / mask_file 映射表
├── studies/
│   ├── CT-0/                     # study_0001 ~ study_0254（无掩码）
│   ├── CT-1/                     # study_0255 ~ study_0304（带掩码，本项目训练用）
│   ├── CT-2/
│   ├── CT-3/
│   └── CT-4/
└── masks/                        # study_0255_mask.nii.gz ~ study_0304_mask.nii.gz
```

### 3.2 数据特点
- **模态**：CT，Hounsfield 单位（约 -1024 ~ 3000）
- **格式**：3D `.nii.gz` 体数据，轴位切片数约 30-64 张/例
- **掩码**：二值 {0=背景, 1=病灶}，前景像素占比约 0.5%
- **带掩码样本**：仅 CT-1 类别的 50 例（study_0255–0304）

### 3.3 预处理（在 dataloader 中完成）
本项目 dataloader（`guided_diffusion/mosmedloader.py`）在 `__getitem__` 中实时完成预处理，无需单独预处理步骤：
1. **HU 窗口**：肺窗 L=-600 / W=1500，裁剪到 [-1350, 150] 并归一化到 [0, 1]
2. **轴位切片**：3D 体数据沿 z 轴切片为 2D 图像
3. **空切片过滤**：`skip_empty=True` 仅保留有病灶的切片（50 例共约 785 张病灶切片）
4. **二值化**：掩码 > 0 → 1，其余 → 0

> 如需预切片以加速训练（可选，非必需），运行 `scripts/prepare_mosmed_slices.py`。

---

## 4. 代码结构

### 4.1 新增文件
| 文件 | 作用 |
|------|------|
| `guided_diffusion/mosmedloader.py` | MosMed 数据加载器（3D nii.gz → 2D 切片） |
| `scripts/mosmed_eval.py` | MosMed 专用评估脚本（Dice/IoU + 阈值搜索） |
| `scripts/prepare_mosmed_slices.py` | 可选：预切片到 2D nii.gz（加速 IO） |
| `README_MOSMED.md` | 本文档 |

### 4.2 修改文件
| 文件 | 修改内容 |
|------|---------|
| `scripts/segmentation_train.py` | 注册 `MOSMED` 分支，设置 `in_ch=2` |
| `scripts/segmentation_sample.py` | 注册 `MOSMED` 分支；添加 `--max_samples`；修复 staple 反相问题 |
| `guided_diffusion/script_util.py` | 暴露 `loss_type` 及 BCE-Dice 超参 |
| `guided_diffusion/gaussian_diffusion.py` | 实现 BCE+Tversky 区域损失；修复 V2 UNet 推理 2 通道塌陷 |
| `guided_diffusion/train_util.py` | 修复单机 dist 守卫；修复 `find_ema_checkpoint` 文件名匹配 |

---

## 5. 快速开始

> 以下命令均在**项目根目录**下执行（Windows 示例，Linux/macOS 同理）。MosMed 数据集请放在项目根目录下，文件名保持默认。

### 5.1 训练

```cmd
cd scripts
python segmentation_train.py --data_name MOSMED --data_dir ../MosMedData-Chest-CT-Scans-with-COVID-19-Related-Findings-main --image_size 256 --num_channels 128 --in_ch 2 --version new --loss_type bce_dice --bce_pos_weight 200 --tversky_alpha 0.3 --tversky_beta 0.7 --batch_size 2 --lr 1e-4 --save_interval 1000 --log_interval 100 --out_dir ./results/mosmed_bcedice_v2
```

### 5.2 推理

```cmd
python scripts/segmentation_sample.py --data_name MOSMED --data_dir ./MosMedData-Chest-CT-Scans-with-COVID-19-Related-Findings-main --model_path ./scripts/results/mosmed_bcedice_v2/savedmodel073000.pt --out_dir ./scripts/results/mosmed_pred_v2_73k --max_samples 20 --image_size 256 --num_channels 128 --in_ch 2 --version new --loss_type bce_dice --batch_size 1 --num_ensemble 1
```

- `--max_samples 20`：只推理 20 张（快速评估）；设为 0 推理全部
- `--num_ensemble 1`：单样本推理（约 60 秒/张）；设为 5 为集成推理
- 输出：`{study_id}_slice{idx}_output_ens.jpg`（灰度图，0=背景，255=前景）

### 5.3 评估

```cmd
python scripts/mosmed_eval.py --pred_dir ./scripts/results/mosmed_pred_v2_73k --data_root ./MosMedData-Chest-CT-Scans-with-COVID-19-Related-Findings-main --image_size 256
```

输出示例：
```
==== MosMed evaluation ====
  all slices (n=20, 0 GT-empty): mean Dice=0.2896 best=0.2973 IoU=0.1843
```

> Windows 用户也可直接运行 `scripts\run_infer_eval_mosmed.bat`，一键完成推理+评估。

---

## 6. 关键技术修复

原始 MedSegDiff 框架针对 BraTS（MRI，多模态）和 ISIC（皮肤镜，RGB）设计，直接用于 MosMed（CT，单模态，二值病灶）会遇到多个问题。以下是本次项目的核心修复：

### 6.1 数据加载器：CT 窗口 + 轴位切片
- **问题**：BraTS 数据已是归一化的多模态 MRI；CT 是 Hounsfield 单位，需要窗口化
- **修复**：`mosmedloader.py` 实现肺窗（L=-600, W=1500）裁剪归一化，3D 体数据按轴位切片
- **空切片过滤**：测试模式也启用 `skip_empty`，确保评估只在有病灶的切片上进行

### 6.2 训练损失：从 MSE 改为 BCE+Tversky
- **问题**：原 `BCE_DICE` 损失是桩代码（返回常数）；MSE 损失在前景占比 0.5% 时让模型学成全背景
- **修复**：在 `gaussian_diffusion.py` 的 `training_losses_segmentation` 中实现真正的区域损失：
  - 将扩散噪声 ε 转换为 pred_xstart（预测的分割图）
  - **加权 BCE**：`pos_weight=200` 抵消 0.5% 的前景占比
  - **Tversky 损失**：`α=0.3, β=0.7`，对假阴性（漏检）惩罚更重
- **关键发现**：`zero_module` 初始化的输出层前 1-2 步输出为 0，损失需 warm-up，不可误判为卡死

### 6.3 推理：V2 UNet 2 通道输出塌陷
- **问题**：V2 框架的 `UNetModel_newpreview` 即使 `learn_sigma=False` 也输出 2 通道，但 `x_t` 是 1 通道，`_predict_xstart_from_eps` 断言失败
- **修复**：在 `p_mean_variance()` 中进入 xstart 预测前添加 `model_output[:, :1]` 塌陷

### 6.4 推理：预测图反相（全白）修复
- **问题**：扩散模型输出在 [-1, 1]（-1=背景, +1=前景）。`staple()` 集成函数内部执行 `s * mvres`，对数值取平方——(-1)²=1 把背景变成了前景，输出全白
- **修复**：保存前映射 [-1,1]→[0,1]（`(x+1)/2`）；`num_ensemble=1` 时跳过 staple

### 6.5 评估：假 Dice 与噪声放大修复
- **问题 1**：`smooth=1e-6` 使空集 vs 空集得 (0+1e-6)/(0+1e-6)=1.0，假完美分
- **修复 1**：pred 和 gt 均空时返回 0.0
- **问题 2**：`pred / pred.max()` 在近全黑预测上把噪声放大到 1.0
- **修复 2**：改用 `pred / 255.0` 固定缩放（`save_image` 已将 [0,1] 映射到 [0,255]）

---

## 7. 关键超参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--image_size` | 256 | 切片分辨率 |
| `--num_channels` | 128 | UNet 基础通道数 |
| `--in_ch` | 2 | 1 CT 通道 + 1 噪声通道 |
| `--version` | new | V2 框架（带 Transformer 引导） |
| `--loss_type` | bce_dice | BCE + Tversky 区域损失 |
| `--bce_pos_weight` | 200 | 正样本权重，≈ 1/前景占比 |
| `--tversky_alpha` | 0.3 | 假阴性权重 |
| `--tversky_beta` | 0.7 | 假阳性权重 |
| `--batch_size` | 2 | 受显存限制 |
| `--lr` | 1e-4 | Adam 学习率 |
| `--save_interval` | 1000 | 每 1000 步存档一次 |

---

## 8. 复现完整流程

从零到获得 Dice=0.29 的完整步骤：

1. **环境**：`pip install -r requirement.txt`
2. **数据**：下载 MosMedData 到项目根目录（见第 3 节）
3. **训练**：执行第 5.1 节命令，训练至 73k 步（`Ctrl+C` 可随时停止，已保存的 checkpoint 不丢失）
4. **推理**：执行第 5.2 节命令（使用 `savedmodel073000.pt`）
5. **评估**：执行第 5.3 节命令，获得平均 Dice≈0.29

---

## 9. 后续改进方向

- **延长训练**：当前 73k 步仍在上升期，训练到 100k-150k 步预期 Dice 可达 0.40-0.50
- **EMA checkpoint**：尝试 `emasavedmodel_0.9999_073000.pt` 推理，EMA 权重通常更平滑
- **多阈值集成**：`num_ensemble=5` 集成推理（已修复 staple 问题）
- **数据增强**：随机翻转、旋转，扩充 50 例有限数据
- **3D 体评估**：当前为 2D 切片级评估，可重建 3D 体积后做体级 Dice

---

## 10. 参考文献

- MedSegDiff-V2: Diffusion based Medical Image Segmentation with Transformer (AAAI 2024)
- MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model (MIDL 2023)
- MosMedData: Chest CT Scans with COVID-19 Related Findings

---

## 致谢

本项目基于 [MedSegDiff](https://github.com/WuJunde/MedSegDiff) 框架（作者：Wu Junde 等，AAAI 2024）进行二次开发，原始框架文档见 [README_ORIGINAL.md](README_ORIGINAL.md)。本仓库的所有适配工作（MosMed 数据加载器、BCE+Tversky 损失重写、推理反相修复、评估脚本等）均为本项目的独立贡献。

感谢 MosMedData 团队提供 COVID-19 胸部 CT 数据集。
