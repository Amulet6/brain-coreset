# 视觉认知工程考查 — 题目2：脑启发核心集选择

## 项目元信息

- **课程**: 华中科技大学 人工智能与自动化学院 "视觉认知工程" 2025-2026 第二学期
- **选题**: 题目2 — 基于脑启发核心集选择的轻量级 VLA 机械臂动作预测
- **截止日期**: 2026年6月20日

## 核心目标

设计并实现 **BRAIN-Coreset** 多阶段算法，从 ALOHA 机器人数据集中智能筛选 10% 核心集，在动作预测上显著优于随机 10% 基线。

## 技术栈

- **语言**: Python 3.10
- **深度学习**: PyTorch 2.6+cu124, torchvision (GPU: RTX 4060 8GB)
- **视觉特征**: Frozen CLIP ViT-B-32 (LAION-2B), 512d → 时序堆叠 3帧 → 1536d
- **语言特征**: CLIP Text Encoder, 固定指令 "transfer the red cube...", 512d
- **VLA 输入**: 1536d (视觉时序) + 512d (语言) = 2048d
- **数据集**: lerobot/aloha_sim_transfer_cube_human (HuggingFace, hf-mirror.com)
- **环境**: conda env `vla` (Python 3.10, CUDA 12.4)

## 模型架构 (最终版 v5)

```
输入: [v_{t-2}, v_{t-1}, v_t, lang_embedding] = 2048d

DualHeadMLP: 共享层 (2048→256→128, ReLU+Dropout)
  ├─ reg_head: Linear(128→6) → 6-DoF 关节角度 (MSE Loss)
  └─ cls_head: Linear(128→1) + Sigmoid → 夹爪开/合 (BCE Loss)
```

## 代码架构

```
├── data/                          # 数据集缓存
├── src/
│   ├── data_loader.py             # ALOHA 数据加载
│   ├── feature_extractor.py       # CLIP 视觉+文本编码器
│   ├── temporal_stack.py          # 时序堆叠 + 语言拼接
│   ├── mlp_model.py               # MLP + DualHeadMLP + 训练
│   ├── stage1_predictive_coding.py # 阶段1: 预测编码
│   ├── stage2_ras_events.py       # 阶段2: RAS 事件
│   ├── stage3_facility_location.py # 阶段3: Facility Location
│   └── visualize_cn.py            # 可视化
├── experiments/
│   ├── run_baseline.py            # Baseline
│   └── run_full_pipeline.py       # 完整流水线
├── run_final_experiment.py        # 最终 VLA 实验
├── report/
│   ├── main.tex                   # LaTeX 报告
│   └── figures/                   # 图表
├── CLAUDE.md                      # 本文件
├── PROJECT_PLAN.md                # 实施方案
├── PROJECT_STATUS.md              # 状态汇报
├── README.md                      # GitHub README
├── acceptance_criteria.md         # 验收清单
└── requirements.txt               # 依赖
```

## 关键设计决策

1. **乘积核**: `S(i,j) = K_v(v_i,v_j) × K_a(a_i,a_j)` — "与"关系
2. **PCA 降维**: 512d → 32d，仅相似度计算时使用
3. **带宽估计**: `σ² = median({||x_i - x_j||²})` 在 5000 随机对上
4. **预算漏斗**: 阶段1+2 并集 → 候选池 ~40% → 阶段3 精选 → 10%
5. **全量候选池**: 不截断 (v5 修复, 核矩阵 ~187MB 完全可接受)
6. **夹爪分类**: DualHeadMLP, BCE Loss + Sigmoid, 评估用 Accuracy/F1
7. **语言模态**: CLIP Text Encoder, 拼接到视觉特征
8. **Multi-Crop TTA**: 原图 + 水平翻转 + 中心裁剪

## 编码规范

- 因果隔离: 阶段1 严格只用 t-3, t-2, t-1 预测 t
- 数据隔离: 训练/测试按 episode 划分 (40/10)
- 模块解耦: 每个 stage_*.py 独立可运行
- 可复现: 所有随机种子固定为 42
- 注释标准: 每个函数有 docstring

## 运行命令

```bash
conda run -n vla python run_final_experiment.py
conda run -n vla python src/visualize_cn.py
```
