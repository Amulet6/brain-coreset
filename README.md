# BRAIN-Coreset: Brain-Inspired Coreset Selection for Lightweight VLA Action Prediction

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green)](https://developer.nvidia.com/cuda-toolkit)

> **华中科技大学 人工智能与自动化学院 — "视觉认知工程" 2025-2026 第二学期**
>
> 选题: 题目2 — 基于脑启发核心集选择的轻量级 VLA 机械臂动作预测

---

## 项目简介

**BRAIN-Coreset** 是一个受大脑认知机制启发的多阶段数据修剪算法，从 ALOHA 机器人数据集中智能筛选 **10% 核心集**。算法借鉴三种脑神经机制：

| 脑机制 | 算法阶段 | 功能 |
|--------|---------|------|
| 预测编码 (Predictive Coding) | 阶段1: 时序过滤 | 保留原型帧, 剔除离群噪声 |
| 网状激活系统 (RAS) | 阶段2: 事件检测 | 捕捉夹爪接触/加速度峰值 |
| 海马模式分离 (Pattern Separation) | 阶段3: 子模优化 | 乘积核 + Facility Location 多样性采样 |

**BRAIN-Coreset** is a multi-stage brain-inspired data pruning algorithm. It intelligently selects the top 10% of frames from ALOHA using predictive coding, RAS event detection, and Facility Location submodular optimization.

---

## 算法架构

```
ALOHA 数据集 (50 Ep, 20,000 帧)
       │
       ▼
CLIP ViT-B-32 (视觉 512d × 3帧) + Text Encoder (语言 512d) = 2048d
       │
  ┌────┴────┐
  ▼         ▼
阶段1       阶段2
预测编码     RAS 事件
(原型采样)   (夹爪/加速度/接触)
  │         │
  └────┬────┘
       ▼
  候选池 (≈40%)
       │
       ▼
阶段3: Facility Location 子模优化
  乘积核 S = K_v × K_a
  Lazy Greedy (1-1/e 保证)
       │
       ▼
  核心集 (10% = 1,600 帧)
       │
       ▼
DualHead MLP: 6-DoF 回归 + 夹爪分类
       │
       ▼
  MSE (6关节) + Accuracy/F1 (夹爪)
```

---

## 实验结果

### 核心指标 (v5 最终版, 2048d VLA)

| 方法 | MSE_6d | Gripper Acc | Gripper F1 |
|------|--------|-------------|------------|
| Full 100% | 0.772 | 61.45% | 0.397 |
| Random 10% | 0.863 | 59.79% | 0.368 |
| **BRAIN-Coreset 10%** | **0.822** | **61.06%** | **0.451** |
| **vs Random** | **+4.7%** | +1.27% | **+22.6%** |

### 迭代历史

| 版本 | 编码器 | 输入 | 关键改进 |
|------|--------|------|---------|
| v1 | ResNet-18 | 512d | 高误差采样 (-6.0%, 失败) |
| v2 | ResNet-18 | 512d | 原型采样 (+0.5%, 特征退化) |
| v3 | CLIP 单帧 | 512d | CLIP 解锁 (+3.1%) |
| v4 | CLIP 时序 | 1536d | 时序堆叠 (+7.1%) |
| **v5** | **CLIP VLA** | **2048d** | **+语言 + 夹爪分类** |

---

## 快速开始

### 环境

```bash
conda create -n vla python=3.10 -y
conda activate vla
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com  # 国内用户
```

### 复现各版本

| 版本 | 编码器 | 采样策略 | 输入维度 | MLP 输出 |
|------|--------|---------|---------|---------|
| v1 | ResNet-18 | 高误差 (`keep_high_error=True`) | 512d | 7-DoF MSE |
| v2 | ResNet-18 | 原型 (`keep_high_error=False`) | 512d | 7-DoF MSE |
| v3 | CLIP 单帧 | 原型 | 512d | 7-DoF MSE |
| v4 | CLIP 时序×3 | 原型 | 1536d | 7-DoF MSE |
| v5 | CLIP VLA | 原型 | 2048d | 6-DoF MSE + 夹爪 BCE |

```bash
# 一键消融: 自动跑通 v1→v5, 结果保存至 data/cache/ablation_results.npy
python experiments/run_ablation.py

# 仅跑 v5 (语言模态 + 双头 MLP)
python run_final_experiment.py

# 单独跑某版本 (修改脚本内 encoder_type/keep_high_error)
python experiments/run_baseline.py          # encoder_type='resnet'|'clip'
python experiments/run_full_pipeline.py     # 完整三阶段, CLIP 时序

# 生成 Fig 1-7 (PNG + PDF)
python src/visualize_cn.py
```

---

## 项目结构

```
├── src/
│   ├── data_loader.py              # ALOHA 数据加载 + 视频解码
│   ├── feature_extractor.py        # ResNet-18 / CLIP 视觉+文本编码器
│   ├── mlp_model.py                # MLP (v1-v4) + DualHeadMLP (v5)
│   ├── temporal_stack.py           # 时序堆叠 + 语言拼接
│   ├── stage1_predictive_coding.py # 阶段1: 预测编码 (高误差/原型)
│   ├── stage2_ras_events.py        # 阶段2: RAS 关键事件检测
│   ├── stage3_facility_location.py # 阶段3: Facility Location 子模优化
│   └── visualize_cn.py            # 可视化 Fig 1-7
├── experiments/
│   ├── run_ablation.py             # ★ 一键消融 v1→v5 (所有版本)
│   ├── run_baseline.py             # 随机基线 (可指定 encoder_type)
│   └── run_full_pipeline.py        # 完整三阶段流水线 (v4)
├── run_final_experiment.py         # ★ v5 最终实验
├── report/figures/                 # 图表 (PNG + PDF, 7张)
├── CLAUDE.md                       # 开发规范
├── PROJECT_PLAN.md                 # 实施方案
├── PROJECT_STATUS.md               # 项目状态汇报
├── acceptance_criteria.md          # 验收清单
└── README.md                       # 本文件
```

## 参考文献

1. Zhao et al. *Learning Fine-Grained Bimanual Manipulation (ACT).* RSS, 2023.
2. Sorscher et al. *Beyond neural scaling laws: beating power law scaling via data pruning.* NeurIPS, 2022.
3. Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model.* arXiv:2406.09246, 2024.
4. Millidge et al. *Predictive coding: a theoretical and experimental review.* arXiv:2107.12979, 2021.

---

*Made for Cognitive Engineering @ HUST*
