# 视觉认知工程考查 — 题目2：脑启发核心集选择

## 项目元信息

- **课程**: 华中科技大学 人工智能与自动化学院 "视觉认知工程" 2025-2026 第二学期
- **试卷**: 2026春季认知工程考查试卷(A卷)
- **选题**: 题目2 — 基于脑启发核心集选择的轻量级 VLA 机械臂动作预测
- **截止日期**: 2026年6月20日

## 核心目标

设计并实现 **BRAIN-Coreset** 多阶段算法，从 ALOHA 机器人数据集中智能筛选 10% 核心集，在动作预测 MSE 上显著优于随机 10% 基线。

## 技术栈

- **语言**: Python 3.8+
- **深度学习**: PyTorch >= 2.0 (CUDA 12.7), torchvision (GPU: RTX 4060 8GB)
- **数据集**: lerobot/aloha_sim_transfer_cube_human (HuggingFace)
- **特征提取**: Frozen ResNet-18 (ImageNet 预训练, Multi-Crop TTA)
- **数值计算**: numpy, scipy, scikit-learn
- **可视化**: matplotlib, seaborn
- **环境管理**: conda env `vla` (Python 3.10, CUDA 12.4)

## 代码架构

```
├── data/                          # 数据集缓存
├── src/
│   ├── feature_extractor.py       # ResNet-18 特征提取 + Multi-Crop TTA
│   ├── stage1_predictive_coding.py # 阶段1: 预测编码时序过滤
│   ├── stage2_ras_events.py       # 阶段2: RAS 关键事件检测
│   ├── stage3_facility_location.py # 阶段3: Facility Location 子模优化
│   ├── stage4_gradient_refine.py  # 阶段4: 梯度精炼 (可选加分)
│   ├── mlp_model.py               # MLP 模型定义 + 训练 + 评估
│   ├── evaluate.py                # 统一评估框架 (MSE, MAE, 各维度)
│   └── visualize.py               # 所有可视化 (Fig 1-7)
├── experiments/
│   ├── run_baseline.py            # 随机 10% 基线
│   ├── run_ablation.py            # 消融实验
│   └── run_full_pipeline.py       # 完整 BRAIN-Coreset 流水线
├── report/
│   └── figures/                   # 报告图表输出
├── CLAUDE.md                      # 本文件
├── PROJECT_PLAN.md                # 详细实施方案与验收标准
├── acceptance_criteria.md         # 逐阶段验收清单
└── requirements.txt               # Python 依赖
```

## 编码规范

### 必须遵守
1. **因果隔离**: 阶段1 严格只用 t-3, t-2, t-1 预测 t，禁止任何未来信息泄漏
2. **数据隔离**: 训练/测试按 episode 划分，禁止同 episode 跨集合
3. **模块解耦**: 每个 stage_*.py 独立可运行，不依赖全局状态
4. **可复现**: 所有随机种子固定为 42，脚本内显式设 `torch.manual_seed(42)`
5. **注释标准**: 每个函数必须有 docstring，关键算法行有行内注释说明对应脑机制
6. **特征规范**: `v_i` = L2-normalized 512d ResNet-18 特征; `a_i` = L2-normalized 7d 动作

### 禁止事项
- 禁止在阶段1构建线性模型时使用目标帧附近的样本拟合参数
- 禁止在阶段2使用布尔值直接判断夹爪开闭
- 禁止在阶段3使用原始 512d 特征直接计算欧氏距离（必须先 PCA 降维到 32d）
- 禁止任何形式的 train-test leak

## 关键设计决策 (已确认，不可随意更改)

1. **乘积核**: `S(i,j) = K_v(v_i,v_j) × K_a(a_i,a_j)` — "与"关系，非加权和
2. **PCA 降维**: 512d → 32d，仅在相似度计算时使用，MLP 训练保留 512d
3. **带宽估计**: `σ² = median({||x_i - x_j||²})` 在 5000 随机对上计算
4. **预算漏斗**: 阶段1+2 并集 → 候选池 25% → 阶段3 精选 → 核心集精确 10%
5. **阈值控制**: 二分搜索 k 使候选池精确在 25% ± 2%
6. **阶段4 降级**: 仅作为可选加分项，不纳入主实验流程
7. **Multi-Crop TTA**: 原图 + 水平翻转 + 中心裁剪 → 平均特征

## 联动更新规则

> **任何代码改动后，必须检查并同步更新以下文件：**
> - `CLAUDE.md` — 架构、命令、规范变更
> - `PROJECT_PLAN.md` — 算法设计、实验条件变更
> - `acceptance_criteria.md` — 验收标准变更
> - `memory/` 对应文件 — 跨会话持久化

## 文件同步要求

- `requirements.txt` 是依赖的唯一真相源，环境变更必须反映于此
- 所有 .py 文件的 import 列表与 requirements.txt 保持一致
- 实验结果数据存储路径统一在 `experiments/` 下

## README.md

- **最后生成**，项目完成 + 报告定稿后写
- 内容：中英双语、项目简介、算法架构图、快速开始、实验结果摘要
- 目标：传到 GitHub 上可直接展示

## 报告标准

- PDF 格式，LaTeX 排版风格（matplotlib 图表统一字体、字号）
- 页数 7-8 页正文 + 附录（源码清单）
- 所有图表嵌入正文对应位置，图号连续
- 参考文献 ≥ 4 篇，格式统一
- 中文正文，专业术语保留英文原名

## 运行命令

```bash
# 激活环境
conda run -n vla python <script>

# 安装缺失依赖
pip install datasets lerobot

# 下载数据集 (首次运行自动)
conda run -n vla python experiments/run_baseline.py

# 运行完整流水线
conda run -n vla python experiments/run_full_pipeline.py

# 运行消融实验
conda run -n vla python experiments/run_ablation.py
```
