# BRAIN-Coreset: Brain-Inspired Coreset Selection for Lightweight VLA Action Prediction

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **华中科技大学 人工智能与自动化学院 — "视觉认知工程" 2025-2026 第二学期课程考查**
>
> 选题: 题目2 — 基于脑启发核心集选择的轻量级 VLA 机械臂动作预测

---

## 📖 项目简介 | Overview

**BRAIN-Coreset** 是一个受大脑认知机制启发的多阶段数据修剪算法，用于从 ALOHA 机器人数据集中智能筛选 **10% 核心集**。该算法借鉴了三种脑神经机制：

| 脑机制 | 对应算法 | 功能 |
|--------|---------|------|
| **预测编码** (Predictive Coding) | 阶段1: 时序过滤 | 保留原型帧, 剔除离群噪声 |
| **网状激活系统** (RAS) | 阶段2: 事件检测 | 捕捉夹爪接触/加速度峰值等关键瞬间 |
| **海马模式分离** (Pattern Separation) | 阶段3: 子模优化 | 乘积核 + Facility Location 多样性采样 |

在仅使用 **10% 训练数据** 的条件下, BRAIN-Coreset 在动作预测任务上的 MSE 比随机采样基线 **降低 7.1%**, 且所有 7 个动作维度均获得正改善。

**BRAIN-Coreset** is a multi-stage brain-inspired data pruning algorithm that intelligently selects a 10% coreset from the ALOHA robot dataset. It reduces action prediction MSE by **7.1%** compared to random sampling, with positive improvements across all 7 action dimensions.

---

## 🧠 算法架构 | Algorithm Pipeline

```
ALOHA 数据集 (50 Episodes, 20,000 Frames)
         │
         ▼
  Frozen CLIP ViT-B-32 特征提取 (512d × 3帧时序 = 1536d)
         │
    ┌────┴────┐
    ▼         ▼
阶段1:       阶段2:
预测编码      RAS 事件检测
(原型采样)    (夹爪/加速度/接触)
    │         │
    └────┬────┘
         ▼
    候选池 (~40% 全量)
         │
         ▼
阶段3: Facility Location 子模优化
    乘积核: S = K_v × K_a
    Lazy Greedy (1-1/e 保证)
         │
         ▼
    核心集 (精确 10% = 1,600 帧)
         │
         ▼
    MLP 动作预测 (1536 → 256 → 128 → 7)
         │
         ▼
    MSE 评估 vs 随机基线
```

---

## 📊 实验结果 | Results

### 核心指标

| 方法 | MSE | vs Random |
|------|-----|-----------|
| Full 100% (理论上界) | 4.766 | — |
| Random 10% (基线) | 6.098 | 基准 |
| **BRAIN-Coreset 10%** | **5.662** | **+7.1%** ✅ |

### 各维度改善

| 维度 | Random | BRAIN-Coreset | 改善 |
|------|--------|---------------|------|
| 腕部角度 | 0.720 | 0.623 | **+13.5%** |
| 腕部转动 | 0.943 | 0.838 | **+11.2%** |
| 前臂旋转 | 0.519 | 0.462 | **+10.9%** |
| 腰部 | 0.887 | 0.798 | **+10.0%** |
| 肘部 | 0.924 | 0.867 | +6.2% |
| 肩部 | 1.077 | 1.055 | +2.0% |
| 夹爪 | 1.029 | 1.020 | +0.8% |

### 特征编码器演进

| 阶段 | 编码器 | 输入 | 改善 | 关键发现 |
|------|--------|------|------|---------|
| v1 | ResNet-18 | 512d | -6.0% | 高误差采样 = 保留离群点 → 失败 |
| v2 | ResNet-18 | 512d | +0.5% | 特征退化, 数据天花板仅 0.19% |
| v3 | CLIP 单帧 | 512d | +3.1% | CLIP 解锁特征空间 |
| v4 | **CLIP 时序** | **1536d** | **+7.1%** | 时序上下文 + 原型采样 |

---

## 🚀 快速开始 | Quick Start

### 环境要求

- Python 3.10+
- CUDA 12.x (可选, CPU 也可运行)
- 8GB+ RAM

### 安装

```bash
# 克隆仓库
git clone https://github.com/<your-username>/brain-coreset.git
cd brain-coreset

# 创建虚拟环境
conda create -n vla python=3.10 -y
conda activate vla

# 安装依赖
pip install -r requirements.txt

# 设置 HuggingFace 镜像 (国内用户)
export HF_ENDPOINT=https://hf-mirror.com
```

### 运行实验

```bash
# 1. Baseline: 随机 10% 采样
python experiments/run_baseline.py

# 2. 完整 BRAIN-Coreset 流水线
python experiments/run_full_pipeline.py

# 3. 生成报告图表
python src/visualize_cn.py
```

首次运行会自动下载 ALOHA 数据集 (~200MB) 和 CLIP 模型 (~600MB)，请确保网络畅通。

---

## 📁 项目结构 | Project Structure

```
brain-coreset/
├── src/                               # 源代码
│   ├── data_loader.py                 # ALOHA 数据加载 & 视频解码
│   ├── feature_extractor.py           # ResNet-18 / CLIP 特征提取器
│   ├── mlp_model.py                   # MLP 动作预测模型
│   ├── temporal_stack.py              # 时序特征堆叠 [v_{t-2}, v_{t-1}, v_t]
│   ├── stage1_predictive_coding.py    # 阶段1: 预测编码时序过滤
│   ├── stage2_ras_events.py           # 阶段2: RAS 关键事件检测
│   ├── stage3_facility_location.py    # 阶段3: Facility Location 子模优化
│   └── visualize_cn.py               # 可视化脚本 (中文版)
├── experiments/                       # 实验入口
│   ├── run_baseline.py                # 随机 10% 基线实验
│   └── run_full_pipeline.py           # BRAIN-Coreset 完整流水线
├── report/figures/                    # 报告图表 (PNG + PDF)
├── data/cache/                        # 数据缓存 (自动生成)
├── CLAUDE.md                          # 项目开发规范
├── PROJECT_PLAN.md                    # 详细实施方案
├── PROJECT_STATUS.md                  # 项目进度汇报
├── acceptance_criteria.md             # 逐阶段验收清单
├── requirements.txt                   # Python 依赖列表
└── README.md                          # 本文件
```

---

## 🔬 关键设计决策 | Key Design Decisions

| 决策 | 内容 | 依据 |
|------|------|------|
| **乘积核** | `S(i,j) = K_v(v_i,v_j) × K_a(a_i,a_j)` | 视觉+动作 "与"关系, 避免加权和的超参问题 |
| **PCA 降维** | 512d → 32d (仅在相似度计算时) | 避免高维空间距离同质化 |
| **带宽估计** | `σ² = median(距离²)` | 非参数自适应, 5000随机对计算 |
| **预算漏斗** | 阶段1+2 并集 → 候选池 ~40% → 阶段3 精选 → 10% | 三阶段逐级筛选 |
| **因果隔离** | 阶段1 严格只用 t-3, t-2, t-1 预测 t | 防止未来信息泄漏 |
| **时序堆叠** | [v_{t-2}, v_{t-1}, v_t] → 1536d | 模拟前额叶工作记忆, 赋予运动趋势感知 |

---

## 📚 参考文献 | References

1. Zhao T Z, Kumar V, Levine S, et al. **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT).** RSS, 2023. [[arXiv:2304.13705](https://arxiv.org/abs/2304.13705)]
2. Sorscher B, Geirhos R, Shekhar S, et al. **Beyond neural scaling laws: beating power law scaling via data pruning.** NeurIPS, 2022. [[arXiv:2206.14486](https://arxiv.org/abs/2206.14486)]
3. Kim M J, Pertsch K, Karamcheti S, et al. **OpenVLA: An Open-Source Vision-Language-Action Model.** arXiv:2406.09246, 2024. [[arXiv:2406.09246](https://arxiv.org/abs/2406.09246)]
4. Millidge B, Seth A, Buckley C L. **Predictive coding: a theoretical and experimental review.** arXiv:2107.12979, 2021. [[arXiv:2107.12979](https://arxiv.org/abs/2107.12979)]

---

## 📄 许可 | License

本项目仅用于学术研究与课程考查目的。

---

*Made with ❤️ for Cognitive Engineering @ HUST*
