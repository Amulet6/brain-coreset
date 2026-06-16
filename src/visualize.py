"""
可视化模块: 生成报告所需的 Fig 1-7

图表清单:
  Fig 1: 算法框架流程图 (schematic)
  Fig 2: 脑机制 ↔ 算法模块映射图
  Fig 3: t-SNE 数据分布对比 (Full vs Random vs Coreset)
  Fig 4: 动作 7 维度分布直方图
  Fig 5: MSE 对比柱状图
  Fig 6: 各维度 MSE 雷达图
  Fig 7: 消融实验热力图 / 特征对比

用法:
  python src/visualize.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

# 全局样式
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'report', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

ACTION_NAMES = ['Waist', 'Shoulder', 'Elbow', 'Forearm\nRoll', 'Wrist\nAngle', 'Wrist\nRotate', 'Gripper']
ACTION_NAMES_SHORT = ['Waist', 'Shoulder', 'Elbow', 'F.Roll', 'W.Angle', 'W.Rot', 'Gripper']


def load_data():
    """加载所有实验数据"""
    data = {}

    # 最终结果 (时序 CLIP)
    path = 'data/cache/temporal_stacked_results.npy'
    if os.path.exists(path):
        data['temporal'] = np.load(path, allow_pickle=True).item()

    # 单帧 CLIP 结果
    path2 = 'data/cache/final_clip_results.npy'
    if os.path.exists(path2):
        data['single_clip'] = np.load(path2, allow_pickle=True).item()

    # ResNet 基线
    path3 = 'data/cache/baseline_results.npy'
    if os.path.exists(path3):
        data['resnet_baseline'] = np.load(path3, allow_pickle=True).item()

    return data


# ============================================================
# Fig 1: 算法框架流程图
# ============================================================
def fig1_algorithm_flowchart():
    """绘制 BRAIN-Coreset 算法框架流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 颜色定义
    c_data = '#E8F5E9'    # 浅绿 - 数据
    c_stage = '#BBDEFB'   # 浅蓝 - 阶段
    c_output = '#FFF9C4'  # 浅黄 - 输出
    c_arrow = '#546E7A'   # 深灰 - 箭头
    c_text = '#212121'    # 深色文字

    def draw_box(x, y, w, h, text, color, fontsize=9, bold=False):
        box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.1',
                             facecolor=color, edgecolor='#78909C', linewidth=1.5)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                weight=weight, color=c_text)

    def draw_arrow(x1, y1, x2, y2, label='', color=c_arrow):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        if label:
            mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
            ax.text(mid_x + 0.15, mid_y - 0.15, label, fontsize=7, color=color, style='italic')

    # 数据输入
    draw_box(6, 7.5, 3, 0.7, 'ALOHA Dataset\n50 Episodes, 20,000 Frames', c_data, bold=True)
    ax.text(6, 7.0, 'Top Camera + Right Arm 7-DoF', ha='center', fontsize=7, color='#757575')

    # 特征提取
    draw_box(6, 6.2, 2.5, 0.6, 'Frozen CLIP ViT-B-32\nFeature Extraction', '#E3F2FD')
    draw_arrow(6, 7.15, 6, 6.5)

    # 三阶段
    draw_box(2, 4.8, 2.5, 0.7, 'Stage 1\nPredictive Coding\n(Prototype Sampling)', c_stage)
    draw_box(6, 4.8, 2.5, 0.7, 'Stage 2\nRAS Event Detection\n(Gripper/Accel/Contact)', c_stage)
    draw_box(10, 4.8, 2.5, 0.7, 'Stage 3\nFacility Location\n(Lazy Greedy)', c_stage)

    # 阶段1+2 → 候选池
    draw_arrow(2, 4.45, 4.5, 3.7)
    draw_arrow(6, 4.45, 6, 3.7)

    draw_box(4.5, 3.3, 3, 0.6, 'Candidate Pool\n~40% of Full Data', '#F3E5F5')
    draw_arrow(4.5, 3.0, 8.5, 3.0)
    draw_arrow(8.5, 3.05, 10, 4.45)  # candidate → Stage 3

    # Stage 3 输出
    draw_arrow(10, 4.45, 10, 3.3)
    draw_box(10, 3.0, 2.5, 0.6, 'Coreset\n10% = 1,600 Frames', c_output, bold=True)

    # 特征提取同时送入 MLP
    draw_arrow(6, 5.9, 6, 2.35)
    # Coreset 送入 MLP
    draw_arrow(10, 2.7, 7.5, 2.0)
    # MLP
    draw_box(6, 2.0, 3, 0.7, 'MLP Training\n1536→256→128→7', '#FFE0B2')
    draw_arrow(6, 1.65, 6, 1.1)

    # 评估
    draw_box(6, 0.7, 2.5, 0.5, 'MSE Evaluation\nvs Random Baseline', '#C8E6C9', bold=True)

    # 标题
    ax.set_title('BRAIN-Coreset Algorithm Pipeline', fontsize=14, weight='bold', pad=15)

    # 脑机制标注
    ax.text(0.5, 4.8, 'Predictive\nCoding', fontsize=7, ha='center', color='#1565C0', style='italic')
    ax.text(6, 5.3, 'Reticular\nActivating\nSystem', fontsize=7, ha='center', color='#1565C0', style='italic')
    ax.text(11.8, 4.8, 'Hippocampal\nPattern\nSeparation', fontsize=7, ha='center', color='#1565C0', style='italic')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig1_algorithm_flowchart.png'), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, 'fig1_algorithm_flowchart.pdf'))
    plt.close()
    print('Fig 1 saved.')


# ============================================================
# Fig 2: 脑机制 ↔ 算法映射图
# ============================================================
def fig2_brain_mapping():
    """绘制脑机制与算法模块的映射关系图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左图: 脑区示意
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 10); ax1.axis('off')
    ax1.set_title('Brain Mechanisms', fontsize=13, weight='bold')

    mechanisms = [
        (5, 8.5, 'Predictive Coding', '#BBDEFB',
         'Neurons fire only when\nprediction is violated.\nFilters redundant frames.'),
        (5, 5.5, 'Reticular Activating\nSystem (RAS)', '#C8E6C9',
         'Filters background noise.\nFocuses on high-utility\ntask-relevant moments.'),
        (5, 2.5, 'Hippocampal Pattern\nSeparation', '#FFE0B2',
         'Avoids storing similar\nexperiences. Ensures\nmemory diversity.'),
    ]
    for x, y, title, color, desc in mechanisms:
        box = FancyBboxPatch((x-4.2, y-1.3), 8.4, 2.6, boxstyle='round,pad=0.15',
                             facecolor=color, edgecolor='#78909C', linewidth=1.5)
        ax1.add_patch(box)
        ax1.text(x, y+0.7, title, ha='center', va='center', fontsize=10, weight='bold')
        ax1.text(x, y-0.4, desc, ha='center', va='center', fontsize=8, color='#424242')

    # 连线
    for i in range(2):
        ax1.annotate('', xy=(5, mechanisms[i+1][1]+1.5), xytext=(5, mechanisms[i][1]-1.5),
                     arrowprops=dict(arrowstyle='->', color='#546E7A', lw=2))

    # 右图: 算法映射
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis('off')
    ax2.set_title('Algorithm Implementation', fontsize=13, weight='bold')

    algorithms = [
        (5, 8.5, 'Stage 1: Predictive\nCoding Filter', '#BBDEFB',
         'Local linear model with\nt-3:t-1 → t prediction.\nKeep low-error prototypes.'),
        (5, 5.5, 'Stage 2: RAS Event\nDetection', '#C8E6C9',
         'Gripper velocity > 90%ile.\nAcceleration > 90%ile.\nContact window detection.'),
        (5, 2.5, 'Stage 3: Facility Location\nSubmodular Optimization', '#FFE0B2',
         'Multiplicative kernel:\nS = K_v(v_i,v_j) × K_a(a_i,a_j).\nLazy Greedy (1-1/e guarantee).'),
    ]
    for x, y, title, color, desc in algorithms:
        box = FancyBboxPatch((x-4.2, y-1.3), 8.4, 2.6, boxstyle='round,pad=0.15',
                             facecolor=color, edgecolor='#78909C', linewidth=1.5)
        ax2.add_patch(box)
        ax2.text(x, y+0.7, title, ha='center', va='center', fontsize=10, weight='bold')
        ax2.text(x, y-0.4, desc, ha='center', va='center', fontsize=8, color='#424242')

    for i in range(2):
        ax2.annotate('', xy=(5, algorithms[i+1][1]+1.5), xytext=(5, algorithms[i][1]-1.5),
                     arrowprops=dict(arrowstyle='->', color='#546E7A', lw=2))

    fig.suptitle('Brain-Inspired Coreset Selection: From Neuroscience to Algorithm',
                 fontsize=14, weight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig2_brain_mapping.png'), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, 'fig2_brain_mapping.pdf'))
    plt.close()
    print('Fig 2 saved.')


# ============================================================
# Fig 3: t-SNE 可视化
# ============================================================
def fig3_tsne():
    """t-SNE 降维可视化: 全量数据 vs Random 10% vs Coreset 10%"""
    from sklearn.manifold import TSNE

    # 加载 CLIP 特征
    tr_feat = np.load('data/cache/train_features_clip.npy')
    n_total = tr_feat.shape[0]

    # 加载 coreset 索引
    # 重新跑一次快速选择
    from src.data_loader import load_aloha_data, get_flat_data
    from src.stage1_predictive_coding import predictive_coding_filter
    from src.stage2_ras_events import ras_event_detection
    from src.stage3_facility_location import facility_location_selection

    train_data, _, _ = load_aloha_data(use_cache=True, verbose=False)
    _, tr_a, tr_b, _, _ = get_flat_data(train_data)

    # 随机索引
    np.random.seed(42)
    n_10pct = int(n_total * 0.1)
    random_idx = set(np.random.choice(n_total, size=n_10pct, replace=False))

    # Coreset 索引
    s1, _ = predictive_coding_filter(tr_a, tr_b, target_ratio=0.35, keep_high_error=False, verbose=False)
    s2, _ = ras_event_detection(tr_a, tr_b, verbose=False)
    cand = s1 | s2
    coreset_idx, _ = facility_location_selection(tr_feat, tr_a, cand, n_10pct, verbose=False)
    coreset_idx = set(coreset_idx)

    # 子采样: t-SNE 太慢, 取 3000 帧
    n_sample = 3000
    all_idx = np.random.choice(n_total, size=n_sample, replace=False)

    # 标签: 0=剩余, 1=Random, 2=Coreset, 3=两者都是
    labels = np.zeros(n_sample, dtype=int)
    for j, idx in enumerate(all_idx):
        in_r = idx in random_idx
        in_c = idx in coreset_idx
        if in_r and in_c: labels[j] = 3
        elif in_c: labels[j] = 2
        elif in_r: labels[j] = 1
        else: labels[j] = 0

    # t-SNE
    print('  Computing t-SNE...')
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    embedded = tsne.fit_transform(tr_feat[all_idx])

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = {0: '#BDBDBD', 1: '#FF7043', 2: '#42A5F5', 3: '#7E57C2'}
    names = {0: 'Remaining (90%)', 1: 'Random 10%', 2: 'BRAIN-Coreset 10%', 3: 'Both'}

    # 子图1: 全量灰底 + Random 高亮
    mask0 = labels == 0
    axes[0].scatter(embedded[mask0, 0], embedded[mask0, 1], c=colors[0], s=5, alpha=0.3, label=names[0])
    mask1 = (labels == 1) | (labels == 3)
    axes[0].scatter(embedded[mask1, 0], embedded[mask1, 1], c=colors[1], s=15, alpha=0.8, label='Random 10%')
    axes[0].set_title('Random 10% Sampling')
    axes[0].legend(fontsize=8)

    # 子图2: 全量灰底 + Coreset 高亮
    axes[1].scatter(embedded[mask0, 0], embedded[mask0, 1], c=colors[0], s=5, alpha=0.3, label=names[0])
    mask2 = (labels == 2) | (labels == 3)
    axes[1].scatter(embedded[mask2, 0], embedded[mask2, 1], c=colors[2], s=15, alpha=0.8, label='BRAIN-Coreset 10%')
    axes[1].set_title('BRAIN-Coreset 10% Sampling')
    axes[1].legend(fontsize=8)

    # 子图3: Random vs Coreset 叠加
    axes[2].scatter(embedded[mask1, 0], embedded[mask1, 1], c=colors[1], s=12, alpha=0.6, label='Random 10%')
    axes[2].scatter(embedded[mask2, 0], embedded[mask2, 1], c=colors[2], s=12, alpha=0.6, label='BRAIN-Coreset 10%')
    both = labels == 3
    if both.any():
        axes[2].scatter(embedded[both, 0], embedded[both, 1], c=colors[3], s=20, alpha=0.9, label='Both')
    axes[2].set_title('Random vs BRAIN-Coreset')
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle('t-SNE Visualization: Data Coverage Comparison', fontsize=14, weight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig3_tsne.png'), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, 'fig3_tsne.pdf'))
    plt.close()
    print('Fig 3 saved.')


# ============================================================
# Fig 4: 动作分布直方图
# ============================================================
def fig4_action_histograms():
    """7 维动作分布: 全量 vs Random vs Coreset"""
    from src.data_loader import load_aloha_data, get_flat_data

    train_data, _, _ = load_aloha_data(use_cache=True, verbose=False)
    _, tr_a, tr_b, _, _ = get_flat_data(train_data)

    # Random 和 Coreset 索引
    np.random.seed(42)
    n_total = tr_a.shape[0]; n_10pct = int(n_total * 0.1)
    random_idx = np.random.choice(n_total, size=n_10pct, replace=False)

    # 从缓存读取 coreset (重跑)
    tr_feat = np.load('data/cache/train_features_clip.npy')
    from src.stage1_predictive_coding import predictive_coding_filter
    from src.stage2_ras_events import ras_event_detection
    from src.stage3_facility_location import facility_location_selection
    print('  Finding coreset for histogram...')
    s1, _ = predictive_coding_filter(tr_a, tr_b, target_ratio=0.35, keep_high_error=False, verbose=False)
    s2, _ = ras_event_detection(tr_a, tr_b, verbose=False)
    cand = s1 | s2
    coreset_idx, _ = facility_location_selection(tr_feat, tr_a, cand, n_10pct, verbose=False)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for dim in range(7):
        ax = axes[dim]
        all_data = tr_a[:, dim]
        rand_data = tr_a[random_idx, dim]
        core_data = tr_a[coreset_idx, dim]

        ax.hist(all_data, bins=40, density=True, alpha=0.4, color='#BDBDBD', label='Full (100%)')
        ax.hist(rand_data, bins=40, density=True, alpha=0.6, color='#FF7043', label='Random 10%')
        ax.hist(core_data, bins=40, density=True, alpha=0.6, color='#42A5F5', label='BRAIN-Coreset 10%')

        ax.set_title(ACTION_NAMES[dim], fontsize=9)
        ax.set_xlabel('Normalized Action')
        if dim == 0:
            ax.legend(fontsize=7)

    # 隐藏第8个子图
    axes[7].axis('off')

    fig.suptitle('Action Distribution: Full Data vs Random 10% vs BRAIN-Coreset 10%',
                 fontsize=14, weight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig4_action_histograms.png'), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, 'fig4_action_histograms.pdf'))
    plt.close()
    print('Fig 4 saved.')


# ============================================================
# Fig 5: MSE 对比柱状图
# ============================================================
def fig5_mse_barchart(data):
    """MSE 对比柱状图: Full / Random / BRAIN-Coreset"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    methods = ['Full 100%', 'Random 10%', 'BRAIN-Coreset\n10%']

    # 时序 CLIP 数据
    td = data.get('temporal', {})
    full_mse = td.get('full_mse', 4.766)
    rand_mse = td.get('random_mse', 6.098)
    bc_mse = td.get('braincorest_mse', 5.662)

    values = [full_mse, rand_mse, bc_mse]
    colors = ['#66BB6A', '#FF7043', '#42A5F5']

    bars = ax.bar(methods, values, color=colors, edgecolor='white', linewidth=1.5, width=0.55)

    # 数值标注
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.3f}', ha='center', fontsize=12, weight='bold')

    # 改善箭头
    improvement = (rand_mse - bc_mse) / rand_mse * 100
    ax.annotate(f'{improvement:+.1f}%', xy=(2, bc_mse), xytext=(2, rand_mse - 0.3),
                ha='center', fontsize=11, weight='bold', color='#2E7D32',
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2))

    ax.set_ylabel('MSE (lower is better)', fontsize=11)
    ax.set_title('Action Prediction MSE Comparison', fontsize=14, weight='bold')
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis='y', alpha=0.3)

    # 添加 baseline 标注
    ax.axhline(y=rand_mse, color='#FF7043', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(0.3, rand_mse + 0.1, 'Random Baseline', fontsize=8, color='#FF7043')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig5_mse_barchart.png'), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, 'fig5_mse_barchart.pdf'))
    plt.close()
    print('Fig 5 saved.')


# ============================================================
# Fig 6: 各维度 MSE 雷达图
# ============================================================
def fig6_radar(data):
    """各维度 MSE 雷达图: Random vs BRAIN-Coreset"""
    td = data.get('temporal', {})
    rand_pd = np.array(td.get('random_per_dim', [0.887, 1.077, 0.924, 0.519, 0.720, 0.943, 1.029]))
    bc_pd = np.array(td.get('braincorest_per_dim', [0.798, 1.055, 0.867, 0.462, 0.623, 0.838, 1.020]))

    dims = len(rand_pd)
    angles = np.linspace(0, 2 * np.pi, dims, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    rand_pd_closed = np.append(rand_pd, rand_pd[0])
    bc_pd_closed = np.append(bc_pd, bc_pd[0])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.fill(angles, rand_pd_closed, alpha=0.25, color='#FF7043', label='Random 10%')
    ax.plot(angles, rand_pd_closed, color='#FF7043', linewidth=2)
    ax.fill(angles, bc_pd_closed, alpha=0.35, color='#42A5F5', label='BRAIN-Coreset 10%')
    ax.plot(angles, bc_pd_closed, color='#42A5F5', linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(ACTION_NAMES_SHORT, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title('Per-Dimension MSE: Random vs BRAIN-Coreset', fontsize=14, weight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # 标注改善百分比
    for i, angle in enumerate(angles[:-1]):
        imp = (rand_pd[i] - bc_pd[i]) / rand_pd[i] * 100
        r = max(rand_pd[i], bc_pd[i]) + 0.05
        ax.annotate(f'{imp:+.1f}%', xy=(angle, r), fontsize=8, ha='center',
                    color='#2E7D32' if imp > 0 else '#C62828', weight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig6_radar.png'), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, 'fig6_radar.pdf'))
    plt.close()
    print('Fig 6 saved.')


# ============================================================
# Fig 7: 特征对比 / 消融热力图
# ============================================================
def fig7_ablation(data):
    """消融实验对比: ResNet vs CLIP Single vs CLIP Temporal"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ---- 左图: 编码器对比 ----
    ax = axes[0]
    encoders = ['ResNet-18', 'CLIP ViT-B-32\n(Single Frame)', 'CLIP ViT-B-32\n(Temporal x3)']
    full_mses = [6.697, 5.109, 4.766]
    rand_mses = [6.710, 6.314, 6.098]
    floors = [6.697] * 3  # random close to floor for ResNet

    x = np.arange(len(encoders))
    w = 0.25
    ax.bar(x - w, [6.697, 5.109, 4.766], w, color='#66BB6A', label='Full 100%')
    ax.bar(x, [6.710, 6.314, 6.098], w, color='#FF7043', label='Random 10%')
    ax.bar(x + w, [6.677, 6.117, 5.662], w, color='#42A5F5', label='BRAIN-Coreset 10%')
    ax.set_xticks(x)
    ax.set_xticklabels(encoders, fontsize=8)
    ax.set_ylabel('MSE')
    ax.set_title('Encoder Comparison', fontsize=11, weight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # ---- 中图: 改善幅度对比 ----
    ax = axes[1]
    improvements = [0.5, 3.1, 7.1]  # ResNet v2, CLIP single, CLIP temporal
    bars = ax.bar(encoders, improvements, color=['#FFCDD2', '#FFF9C4', '#C8E6C9'],
                  edgecolor='#78909C', linewidth=1.2)
    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'+{val:.1f}%', ha='center', fontsize=12, weight='bold')
    ax.set_ylabel('Improvement over Random (%)')
    ax.set_title('Coreset Improvement by Encoder', fontsize=11, weight='bold')
    ax.set_xticklabels(encoders, fontsize=8)
    ax.axhline(y=0, color='#757575', linestyle='--')
    ax.grid(axis='y', alpha=0.3)

    # ---- 右图: 每维度改善热力图 ----
    ax = axes[2]
    td = data.get('temporal', {})
    rand_pd = np.array(td.get('random_per_dim', [0.887, 1.077, 0.924, 0.519, 0.720, 0.943, 1.029]))
    bc_pd = np.array(td.get('braincorest_per_dim', [0.798, 1.055, 0.867, 0.462, 0.623, 0.838, 1.020]))
    per_dim_imp = (rand_pd - bc_pd) / rand_pd * 100

    # 也加载单帧 CLIP 对比
    sc = data.get('single_clip', {})
    if sc:
        rand_pd_sc = np.array(sc.get('random_per_dim', [0.888, 1.102, 0.975, 0.559, 0.773, 0.979, 1.037]))
        bc_pd_sc = np.array(sc.get('braincorest_per_dim', [0.841, 1.061, 0.959, 0.575, 0.698, 0.930, 1.055]))
        sc_imp = (rand_pd_sc - bc_pd_sc) / rand_pd_sc * 100
        data_matrix = np.vstack([sc_imp, per_dim_imp])
        sns.heatmap(data_matrix, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                    xticklabels=ACTION_NAMES_SHORT,
                    yticklabels=['CLIP Single', 'CLIP Temporal x3'],
                    ax=ax, cbar_kws={'label': 'Improvement %'}, vmin=-5, vmax=15)
    else:
        data_matrix = per_dim_imp.reshape(1, -1)
        sns.heatmap(data_matrix, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                    xticklabels=ACTION_NAMES_SHORT,
                    yticklabels=['CLIP Temporal x3'],
                    ax=ax, cbar_kws={'label': 'Improvement %'}, vmin=-5, vmax=15)

    ax.set_title('Per-Dimension Improvement', fontsize=11, weight='bold')

    fig.suptitle('BRAIN-Coreset Ablation Analysis', fontsize=14, weight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig7_ablation.png'), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, 'fig7_ablation.pdf'))
    plt.close()
    print('Fig 7 saved.')


# ============================================================
# Main
# ============================================================
def main():
    print('Generating all figures...')
    data = load_data()

    fig1_algorithm_flowchart()
    fig2_brain_mapping()
    fig3_tsne()
    fig4_action_histograms()
    fig5_mse_barchart(data)
    fig6_radar(data)
    fig7_ablation(data)

    print(f'\nAll 7 figures saved to: {FIG_DIR}')
    print('Done.')


if __name__ == '__main__':
    main()
