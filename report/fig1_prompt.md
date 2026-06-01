# BRAIN-Coreset 算法流程图 — AI 生图 Prompt

---

请生成一张学术论文风格的算法框架流程图，白色背景，专业简洁。整体布局为从上到下的漏斗式结构。

## 整体结构

图表分为 5 个层级，从上到下依次为：数据层 → 并行处理层 →汇聚层 → 精选层 → 评估层。

## 各层详细描述

### 第1层（顶部）：数据输入
- 一个绿色圆角矩形框，位于画面最上方居中
- 框内文字："ALOHA Simulation Dataset"（第一行，加粗）
- 框内副文字："50 Episodes × 400 Frames = 20,000 Frames"（第二行，小字灰色）
- 框下方标注："Top Camera + Right Arm 7-DoF"

### 第2层：特征提取
- 一个浅蓝色圆角矩形框，位于数据框正下方
- 向下箭头从数据框指向此框
- 框内文字："Frozen CLIP ViT-B-32 Feature Extraction"
- 框内副文字："512d × Temporal Stack (3 frames) → 1536d"

### 第3层：三阶段并行处理
从特征提取框分出三条向下的箭头，分别指向左、中、右三个并排的圆角矩形框：

**左框（蓝色）**：
- 标题："Stage 1"（加粗）
- 正文："Predictive Coding Filter"
- 小字标注："Prototype Sampling"（楷体/斜体）
- 框上方用小字蓝色标注："← Predictive Coding"

**中框（绿色）**：
- 标题："Stage 2"（加粗）
- 正文："RAS Event Detection"
- 小字标注："Gripper / Acceleration / Contact"（楷体/斜体）
- 框上方用小字蓝色标注："← Reticular Activating System"

**右框（橙色）**：
- 标题："Stage 3"（加粗）
- 正文："Facility Location"
- 小字标注："Submodular Optimization"（楷体/斜体）
- 框上方用小字蓝色标注："← Pattern Separation"

注意事项：三个框大小一致、水平对齐、间距均匀。左框和中框的下方各有一个向下箭头汇聚到一个框；右框的下方箭头留待后续连接。

### 第4层（中间偏下）：候选池汇聚
- 左框和中框各引出一个向下箭头，汇聚到一个紫色圆角矩形框
- 框内文字："Candidate Pool (Union)"
- 小字标注："≈ 40% of Full Data"
- 从此框右侧引出一个向右的箭头，连接到右框（Stage 3）下方

### 第5层：核心集输出
- Stage 3 下方引出一个向下箭头
- 指向一个黄色圆角矩形框
- 框内文字："Final Coreset"（加粗）
- 小字标注："Exactly 10% = 1,600 Frames"

### 第6层（底部）：训练与评估
- 特征提取框（第2层）引出一个贯穿的向下箭头，绕过左侧，直达底部
- 核心集框引出一个向左下方的箭头
- 两箭头汇聚到一个橙色圆角矩形框
- 框内文字："MLP Training"
- 小字标注："1536 → 256 → 128 → 7, 100 epochs"

- MLP 框下方箭头指向最后一个绿色框
- 框内文字："MSE Evaluation vs Random Baseline"

## 配色方案
- 数据/评估框：浅绿色 #E8F5E9
- 特征提取框：浅蓝色 #E3F2FD
- Stage 1：蓝色 #BBDEFB
- Stage 2：绿色 #C8E6C9
- Stage 3：橙色 #FFE0B2
- 候选池：浅紫色 #F3E5F5
- 核心集：浅黄色 #FFF9C4
- MLP：浅橙色 #FFCC80
- 所有边框：深灰色 #78909C，圆角，线宽适中
- 箭头：深灰色 #546E7A，线宽 1.5-2pt

## 文字规范
- 所有英文，无中文
- 标题/框名用 Bold
- 副文本用 Regular 或 Italic
- 字号分级：大标题 14pt → 框标题 10pt → 正文 8-9pt → 注释 7pt
- 脑机制标注用蓝色小字斜体，放在对应阶段框的上方

## 风格要求
- 学术论文风格，简洁干净
- 不要照片、不要图标、不要渐变
- 纯矢量图风格，适合 LaTeX 论文嵌入
- 白底，无背景网格
- 框之间留有足够间距，箭头不穿过任何文字
- 所有文字清晰可读，无重叠无遮挡

## 图标题
图表底部居中：Fig. 1: BRAIN-Coreset Algorithm Pipeline
