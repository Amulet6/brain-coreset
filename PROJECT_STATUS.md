# BRAIN-Coreset 项目终期状态汇报

> 日期: 2026-06-01 | 选题: 题目2 — 脑启发核心集选择 VLA 动作预测
> 状态: ✅ 全部完成，准备撰写报告

---

## 一、最终实验结果 (CLIP ViT-B-32 + 时序堆叠, 5 次试验)

### 1.1 核心对比

```
┌──────────────────────────────────────┬────────────┬──────────────────────┐
│ 实验条件                              │ MSE        │ vs Random 10%        │
├──────────────────────────────────────┼────────────┼──────────────────────┤
│ Full 100% (16,000 帧, 理论上界)        │ 4.766      │ —                    │
│ Random 10% (1,600 帧, 基线)           │ 6.098±0.04 │ 基准                  │
│ BRAIN-Coreset 10% (1,600 帧)          │ 5.662±0.09 │ +7.1% ✅              │
└──────────────────────────────────────┴────────────┴──────────────────────┘
```

**结论：BRAIN-Coreset 以 10% 数据量实现比随机 10% 好 7.1% 的性能，达到全量性能的 84.2%。**

### 1.2 各维度改善

```
维度                 Random 10%    BRAIN-Coreset 10%   改善
──────────────────────────────────────────────────────────────
腰部 (Waist)           0.887         0.798              +10.0%
肩部 (Shoulder)        1.077         1.055               +2.0%
肘部 (Elbow)           0.924         0.867               +6.2%
前臂旋转 (F.Roll)      0.519         0.462              +10.9% ★
腕部角度 (W.Angle)     0.720         0.623              +13.5% ★★
腕部转动 (W.Rot)       0.943         0.838              +11.2% ★
夹爪 (Gripper)         1.029         1.020               +0.8%
──────────────────────────────────────────────────────────────
总 MSE                 6.098         5.662               +7.1%
```

**全部 7 个维度均为正改善，零退化。**

### 1.3 三阶段演进路线

```
实验阶段                    编码器          输入维度    改善幅度    说明
──────────────────────────────────────────────────────────────────────
v1 高误差采样               ResNet-18       512         -6.0%      保留离群点 → 失败
v2 原型采样                 ResNet-18       512         +0.5%      特征退化,天花板 0.19%
v3 原型采样                 CLIP单帧         512         +3.1%      CLIP解锁特征空间
v4 原型采样 + 时序堆叠       CLIP时序        1536        +7.1% ★    时序上下文 + 原型采样
──────────────────────────────────────────────────────────────────────
```

---

## 二、代码完成度

### 2.1 核心模块

| 模块 | 文件 | 行数 | 状态 |
|------|------|------|:--:|
| 数据加载 | `src/data_loader.py` | ~180 | ✅ |
| 特征提取 | `src/feature_extractor.py` | ~210 | ✅ (ResNet + CLIP) |
| MLP 模型 | `src/mlp_model.py` | ~170 | ✅ |
| 时序堆叠 | `src/temporal_stack.py` | ~50 | ✅ |
| Stage 1 | `src/stage1_predictive_coding.py` | ~200 | ✅ (v1高误差 + v2原型) |
| Stage 2 | `src/stage2_ras_events.py` | ~120 | ✅ |
| Stage 3 | `src/stage3_facility_location.py` | ~310 | ✅ (乘积核 + Lazy Greedy优化) |
| 可视化 | `src/visualize.py` | ~550 | ✅ |
| 可视化(中文) | `src/visualize_cn.py` | ~360 | ✅ |

### 2.2 实验脚本

| 实验 | 文件 | 状态 |
|------|------|:--:|
| Baseline | `experiments/run_baseline.py` | ✅ |
| 完整流水线 | `experiments/run_full_pipeline.py` | ✅ |
| 消融实验 | `experiments/run_ablation.py` | ⬜ (数据已通过内联脚本收集) |

### 2.3 报告素材

| 图表 | 文件 | 状态 |
|------|------|:--:|
| Fig 1 算法流程图 | `report/figures/fig1_flowchart.*` | ✅ |
| Fig 2 脑机制映射 | `report/figures/fig2_brain_mapping.*` | ✅ |
| Fig 3 t-SNE 可视化 | `report/figures/fig3_tsne.*` | ✅ |
| Fig 4 动作分布直方图 | `report/figures/fig4_histograms.*` | ✅ |
| Fig 5 MSE 柱状图 | `report/figures/fig5_mse_bar.*` | ✅ |
| Fig 6 雷达图 | `report/figures/fig6_radar.*` | ✅ |
| Fig 7 消融分析 | `report/figures/fig7_ablation.*` | ✅ |

---

## 三、题目要求对照

| 题目要求 | 完成情况 | 详情 |
|---------|:--:|------|
| 随机 10% 基线 | ✅ | MSE 6.098 ± 0.04, 5 次重复 |
| 脑启发核心集选择算法 | ✅ | 三阶段: 预测编码 + RAS + 子模优化 |
| 算法验证 (vs 随机) | ✅ | +7.1%, 全维度正改善 |
| 冻结视觉模型特征提取 | ✅ | CLIP ViT-B-32 + ResNet-18 双编码器对比 |
| MLP 动作预测 | ✅ | 1536 → 256 → 128 → 7 |
| 源代码 + 注释 | ✅ | 模块化, 含 docstring, 中文注释 |
| 研究报告 7-8 页 | ⬜ | 准备撰写 |

---

## 四、项目亮点

1. **脑认知深度**: 每个算法阶段对应具体神经科学机制, 有文献支撑
2. **工程完整性**: ResNet → CLIP → 时序堆叠, 完整的诊断-改进-验证闭环
3. **算法创新**: 乘积核 (K_v × K_a) + 中位数带宽估计 + Lazy Greedy 子模优化
4. **七维度正改善**: 全维度零退化, 最佳维度 +13.5%
5. **科研方法论**: v1(失败) → v2(持平) → v3(+3.1%) → v4(+7.1%) 展示完整迭代

---

## 五、运行命令

```bash
# 环境
conda activate vla

# 安装依赖 (首次)
pip install -r requirements.txt

# Baseline 实验
python experiments/run_baseline.py

# 完整流水线
python experiments/run_full_pipeline.py

# 生成图表
python src/visualize_cn.py
```

---

## 六、存留问题与改进方向

| 问题 | 原因分析 | 改进方向 |
|------|---------|---------|
| gripper 改善微弱 (+0.8%) | 夹爪动作稀疏 (开/关二值化), 空间多样性帮助有限 | 时序注意力机制或专门的事件建模 |
| 候选池需随机下采样 | 内存限制 (5000帧上限) | 使用在线核计算或分块 Lazy Greedy |
| 未做超参调优 | 时间限制 | MLP 架构/学习率/epoch 网格搜索 |
| Stage 4 未实现 | 策略性放弃 (可能反效果) | 在报告中作为 theoretical extension 讨论 |

---

## 七、文件索引

```
visual recognition project/
├── src/                              # 源代码
│   ├── data_loader.py                # 数据加载
│   ├── feature_extractor.py          # 特征提取 (ResNet + CLIP)
│   ├── mlp_model.py                  # MLP 模型
│   ├── temporal_stack.py             # 时序特征堆叠
│   ├── stage1_predictive_coding.py     # 阶段1: 预测编码
│   ├── stage2_ras_events.py          # 阶段2: RAS 事件检测
│   ├── stage3_facility_location.py   # 阶段3: 子模优化
│   └── visualize_cn.py               # 可视化 (中文版)
├── experiments/                      # 实验脚本
│   ├── run_baseline.py               # Baseline
│   └── run_full_pipeline.py          # 完整流水线
├── report/
│   └── figures/                      # 图表 (PNG + PDF, 中英文)
├── data/cache/                       # 数据缓存
├── CLAUDE.md                         # 项目宪章
├── PROJECT_PLAN.md                   # 详细实施方案
├── PROJECT_STATUS.md                 # 本文件
├── acceptance_criteria.md            # 验收清单
├── requirements.txt                  # Python 依赖
└── README.md                         # GitHub README
```
