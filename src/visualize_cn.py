"""
可视化模块 (中文版): 生成报告所需的 Fig 1-7

图表标注使用中文，专有名词保留英文。
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# 全局样式
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'report', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

ACTION_CN = ['腰部', '肩部', '肘部', '前臂旋转', '腕部角度', '腕部转动', '夹爪']
ACTION_SHORT = ['Waist', 'Shoulder', 'Elbow', 'F.Roll', 'W.Angle', 'W.Rot', 'Gripper']


def load_data():
    data = {}
    for name, path in [
        ('temporal', 'data/cache/temporal_stacked_results.npy'),
        ('single_clip', 'data/cache/final_clip_results.npy'),
        ('resnet', 'data/cache/baseline_results.npy'),
    ]:
        if os.path.exists(path):
            data[name] = np.load(path, allow_pickle=True).item()
    return data


# ============================================================
# Fig 1: 算法框架流程图
# ============================================================
def fig1():
    fig, ax = plt.subplots(1, 1, figsize=(13, 8.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 8.5); ax.axis('off')

    def box(x, y, w, h, text, color, fs=9, bold=False):
        b = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.1',
                           facecolor=color, edgecolor='#78909C', linewidth=1.5)
        ax.add_patch(b)
        ax.text(x, y, text, ha='center', va='center', fontsize=fs,
                weight='bold' if bold else 'normal', color='#212121')

    def arrow(x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#546E7A', lw=1.5))
        if label:
            ax.text((x1+x2)/2 + 0.2, (y1+y2)/2 - 0.15, label, fontsize=7, color='#546E7A', style='italic')

    # 数据输入
    box(6.5, 8.0, 3.5, 0.7, 'ALOHA 仿真数据集\n50 个 Episode, 20,000 帧', '#E8F5E9', bold=True)
    ax.text(6.5, 7.5, 'Top 相机 + 右臂 7-DoF 动作', ha='center', fontsize=7, color='#757575')

    # CLIP 特征提取
    box(6.5, 6.7, 3, 0.6, '冻结 CLIP ViT-B-32\n离线提取视觉特征', '#E3F2FD')
    arrow(6.5, 7.65, 6.5, 7.0)

    # 阶段1
    box(2, 5.2, 3, 0.8, '阶段 1: 预测编码过滤\n(原型采样模式)', '#BBDEFB', bold=True)
    # 阶段2
    box(6.5, 5.2, 3, 0.8, '阶段 2: RAS 关键事件检测\n(夹爪/加速度/接触)', '#C8E6C9', bold=True)
    # 阶段3
    box(11, 5.2, 3, 0.8, '阶段 3: Facility Location\n子模优化 (Lazy Greedy)', '#FFE0B2', bold=True)

    # 阶段1,2 → 候选池
    arrow(2, 4.8, 3.5, 3.8)
    arrow(6.5, 4.8, 3.5, 3.8)
    box(3.5, 3.5, 3.5, 0.65, '候选池 (并集)\n约 40% 全量数据', '#F3E5F5')
    arrow(3.5, 3.15, 9.5, 3.15)
    arrow(9.5, 3.2, 11, 4.8)

    # 阶段3输出
    arrow(11, 4.8, 11, 3.5)
    box(11, 3.2, 3, 0.6, '核心集\n精确 10% = 1,600 帧', '#FFF9C4', bold=True)

    # MLP
    arrow(6.5, 6.4, 6.5, 2.5)
    arrow(11, 2.9, 8, 2.2)
    box(6.5, 2.2, 3.5, 0.7, 'MLP 动作预测训练\n(1536 → 256 → 128 → 7)', '#FFCC80')
    arrow(6.5, 1.85, 6.5, 1.2)

    # 评估
    box(6.5, 0.9, 3, 0.5, 'MSE 评估 vs 随机基线', '#C8E6C9', bold=True)

    # 脑机制标注
    for x, txt in [(2, '预测编码\nPredictive Coding'), (6.5, '网状激活系统\nRAS'),
                    (11, '海马模式分离\nPattern Separation')]:
        ax.text(x, 5.7, txt, fontsize=7, ha='center', color='#1565C0', style='italic')

    ax.set_title('图 1: BRAIN-Coreset 算法框架流程图', fontsize=14, weight='bold', pad=15)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig1_flowchart.png'), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, 'fig1_flowchart.pdf'))
    plt.close(); print('图1 已保存')


# ============================================================
# Fig 2: 脑机制 ↔ 算法映射
# ============================================================
def fig2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, title, items in [
        (ax1, '脑认知机制', [
            (5, 8.5, '预测编码 (Predictive Coding)', '#BBDEFB',
             '大脑持续生成预测信号\n仅在预测被打破时激活神经元\n→ 自动过滤"发呆帧"'),
            (5, 5.5, '网状激活系统 (RAS)', '#C8E6C9',
             '根据任务目标过滤背景噪音\n引导注意力聚焦高信息效用瞬间\n→ 捕捉夹爪接触/轨迹突变'),
            (5, 2.5, '海马体模式分离\n(Pattern Separation)', '#FFE0B2',
             '避免存储高度相似的重复经验\n确保记忆库的全面覆盖\n→ 多样性采样, 消除分布冗余'),
        ]),
        (ax2, '算法实现', [
            (5, 8.5, '阶段 1: 预测编码时序过滤', '#BBDEFB',
             '全局线性模型: 用 t-3:t-1 预测 t\n保留低预测误差帧 (原型样本)\n二分搜索阈值控制保留比例'),
            (5, 5.5, '阶段 2: RAS 关键事件检测', '#C8E6C9',
             '夹爪速度 > 90% 分位\n加速度幅值 > 90% 分位\n减速 + 夹爪联动 → 接触窗口'),
            (5, 2.5, '阶段 3: Facility Location\n子模函数最大化', '#FFE0B2',
             '乘积核: S = K_v × K_a\nPCA 降维 512d → 32d\nLazy Greedy (1-1/e 近似保证)'),
        ]),
    ]:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off'); ax.set_title(title, fontsize=13, weight='bold')
        for x, y, t, c, d in items:
            b = FancyBboxPatch((x-4.3, y-1.3), 8.6, 2.6, boxstyle='round,pad=0.15',
                               facecolor=c, edgecolor='#78909C', linewidth=1.5)
            ax.add_patch(b)
            ax.text(x, y+0.7, t, ha='center', va='center', fontsize=10, weight='bold')
            ax.text(x, y-0.4, d, ha='center', va='center', fontsize=8, color='#424242')
        for i in range(2):
            ax.annotate('', xy=(5, items[i+1][1]+1.5), xytext=(5, items[i][1]-1.5),
                        arrowprops=dict(arrowstyle='->', color='#546E7A', lw=2))

    fig.suptitle('图 2: 脑启发核心集选择 — 从神经科学机制到算法实现',
                 fontsize=14, weight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig2_brain_mapping.png'), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, 'fig2_brain_mapping.pdf'))
    plt.close(); print('图2 已保存')


# ============================================================
# Fig 3: t-SNE 可视化
# ============================================================
def fig3():
    from sklearn.manifold import TSNE
    from src.data_loader import load_aloha_data, get_flat_data
    from src.stage1_predictive_coding import predictive_coding_filter
    from src.stage2_ras_events import ras_event_detection
    from src.stage3_facility_location import facility_location_selection

    tr_feat = np.load('data/cache/train_features_clip.npy')
    n_total = tr_feat.shape[0]; n_10pct = int(n_total * 0.1)
    train_data, _, _ = load_aloha_data(use_cache=True, verbose=False)
    _, tr_a, tr_b, _, _ = get_flat_data(train_data)

    np.random.seed(42)
    random_idx = set(np.random.choice(n_total, size=n_10pct, replace=False))
    s1, _ = predictive_coding_filter(tr_a, tr_b, target_ratio=0.35, keep_high_error=False, verbose=False)
    s2, _ = ras_event_detection(tr_a, tr_b, verbose=False)
    coreset_idx, _ = facility_location_selection(tr_feat, tr_a, s1|s2, n_10pct, verbose=False)
    coreset_idx = set(coreset_idx)

    n_sample = 3000
    all_idx = np.random.choice(n_total, size=n_sample, replace=False)
    labels = np.zeros(n_sample, dtype=int)
    for j, idx in enumerate(all_idx):
        in_r = idx in random_idx; in_c = idx in coreset_idx
        labels[j] = 3 if (in_r and in_c) else (2 if in_c else (1 if in_r else 0))

    print('  计算 t-SNE ...')
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    embedded = tsne.fit_transform(tr_feat[all_idx])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {0: '#BDBDBD', 1: '#FF7043', 2: '#42A5F5', 3: '#7E57C2'}
    labels_cn = {0: '剩余 90%', 1: '随机 10%', 2: 'BRAIN-Coreset 10%', 3: '两者重叠'}

    m0 = labels == 0
    for ax, title, hl_mask in [
        (axes[0], '随机 10% 采样', (labels==1)|(labels==3)),
        (axes[1], 'BRAIN-Coreset 10% 采样', (labels==2)|(labels==3)),
        (axes[2], '随机 vs BRAIN-Coreset 叠加', None),
    ]:
        ax.scatter(embedded[m0,0], embedded[m0,1], c=colors[0], s=4, alpha=0.25)
        if hl_mask is not None:
            ax.scatter(embedded[hl_mask,0], embedded[hl_mask,1], c=colors[1] if '随机' in title else colors[2], s=14, alpha=0.8)
        else:
            ax.scatter(embedded[labels==1,0], embedded[labels==1,1], c=colors[1], s=10, alpha=0.5, label='随机 10%')
            ax.scatter(embedded[labels==2,0], embedded[labels==2,1], c=colors[2], s=10, alpha=0.5, label='BRAIN-Coreset 10%')
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle('图 3: t-SNE 数据分布覆盖对比 (CLIP 特征空间)', fontsize=14, weight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig3_tsne.png'), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, 'fig3_tsne.pdf'))
    plt.close(); print('图3 已保存')


# ============================================================
# Fig 4: 动作分布直方图
# ============================================================
def fig4():
    from src.data_loader import load_aloha_data, get_flat_data
    from src.stage1_predictive_coding import predictive_coding_filter
    from src.stage2_ras_events import ras_event_detection
    from src.stage3_facility_location import facility_location_selection

    train_data, _, _ = load_aloha_data(use_cache=True, verbose=False)
    _, tr_a, tr_b, _, _ = get_flat_data(train_data)
    tr_feat = np.load('data/cache/train_features_clip.npy')

    np.random.seed(42); n_total = tr_a.shape[0]; n_10pct = int(n_total * 0.1)
    random_idx = np.random.choice(n_total, size=n_10pct, replace=False)
    print('  寻找核心集 ...')
    s1, _ = predictive_coding_filter(tr_a, tr_b, target_ratio=0.35, keep_high_error=False, verbose=False)
    s2, _ = ras_event_detection(tr_a, tr_b, verbose=False)
    coreset_idx, _ = facility_location_selection(tr_feat, tr_a, s1|s2, n_10pct, verbose=False)

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))
    axes = axes.flatten()

    for dim in range(7):
        ax = axes[dim]
        ax.hist(tr_a[:, dim], bins=40, density=True, alpha=0.35, color='#BDBDBD', label='全量 100%')
        ax.hist(tr_a[random_idx, dim], bins=40, density=True, alpha=0.6, color='#FF7043', label='随机 10%')
        ax.hist(tr_a[coreset_idx, dim], bins=40, density=True, alpha=0.6, color='#42A5F5', label='BRAIN-Coreset 10%')
        ax.set_title(f'{ACTION_CN[dim]} ({ACTION_SHORT[dim]})', fontsize=9)
        ax.set_xlabel('归一化动作值')
        if dim == 0: ax.legend(fontsize=7)
    axes[7].axis('off')

    fig.suptitle('图 4: 动作各维度分布 — 全量数据 vs 随机 10% vs BRAIN-Coreset 10%',
                 fontsize=14, weight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig4_histograms.png'), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, 'fig4_histograms.pdf'))
    plt.close(); print('图4 已保存')


# ============================================================
# Fig 5: MSE 对比柱状图
# ============================================================
def fig5(data):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    td = data.get('temporal', {})
    methods = ['全量 100%', '随机 10%\n(基线)', 'BRAIN-Coreset\n10%']
    values = [td.get('full_mse', 4.766), td.get('random_mse', 6.098), td.get('braincorest_mse', 5.662)]
    colors = ['#66BB6A', '#FF7043', '#42A5F5']
    bars = ax.bar(methods, values, color=colors, edgecolor='white', linewidth=1.5, width=0.55)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.3f}', ha='center', fontsize=13, weight='bold')

    imp = (values[1] - values[2]) / values[1] * 100
    ax.annotate(f'改善 {imp:+.1f}%', xy=(2, values[2]), xytext=(2, values[1] - 0.35),
                ha='center', fontsize=12, weight='bold', color='#2E7D32',
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2))

    ax.set_ylabel('MSE (越小越好)', fontsize=11)
    ax.set_title('图 5: 动作预测 MSE 对比', fontsize=14, weight='bold')
    ax.set_ylim(0, max(values) * 1.2); ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=values[1], color='#FF7043', linestyle='--', alpha=0.4)
    ax.text(0.25, values[1] + 0.1, '随机基线', fontsize=8, color='#FF7043')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig5_mse_bar.png'), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, 'fig5_mse_bar.pdf'))
    plt.close(); print('图5 已保存')


# ============================================================
# Fig 6: 雷达图
# ============================================================
def fig6(data):
    td = data.get('temporal', {})
    rand_pd = np.array(td.get('random_per_dim', [0.887, 1.077, 0.924, 0.519, 0.720, 0.943, 1.029]))
    bc_pd = np.array(td.get('braincorest_per_dim', [0.798, 1.055, 0.867, 0.462, 0.623, 0.838, 1.020]))

    dims = len(rand_pd)
    angles = np.linspace(0, 2*np.pi, dims, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, np.append(rand_pd, rand_pd[0]), alpha=0.25, color='#FF7043', label='随机 10%')
    ax.plot(angles, np.append(rand_pd, rand_pd[0]), color='#FF7043', linewidth=2)
    ax.fill(angles, np.append(bc_pd, bc_pd[0]), alpha=0.35, color='#42A5F5', label='BRAIN-Coreset 10%')
    ax.plot(angles, np.append(bc_pd, bc_pd[0]), color='#42A5F5', linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(ACTION_CN, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title('图 6: 各维度 MSE 雷达图 — 随机 vs BRAIN-Coreset', fontsize=14, weight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    for i, angle in enumerate(angles[:-1]):
        imp = (rand_pd[i] - bc_pd[i]) / rand_pd[i] * 100
        r = max(rand_pd[i], bc_pd[i]) + 0.05
        ax.annotate(f'{imp:+.1f}%', xy=(angle, r), fontsize=8, ha='center',
                    color='#2E7D32' if imp>0 else '#C62828', weight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig6_radar.png'), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, 'fig6_radar.pdf'))
    plt.close(); print('图6 已保存')


# ============================================================
# Fig 7: 消融/特征对比
# ============================================================
def fig7(data):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # 左: 编码器对比
    ax = axes[0]
    enc = ['ResNet-18', 'CLIP ViT-B-32\n(单帧)', 'CLIP ViT-B-32\n(时序堆叠×3)']
    x = np.arange(3); w = 0.25
    ax.bar(x-w, [6.697, 5.109, 4.766], w, color='#66BB6A', label='全量 100%')
    ax.bar(x,   [6.710, 6.314, 6.098], w, color='#FF7043', label='随机 10%')
    ax.bar(x+w, [6.677, 6.117, 5.662], w, color='#42A5F5', label='BRAIN-Coreset 10%')
    ax.set_xticks(x); ax.set_xticklabels(enc, fontsize=8)
    ax.set_ylabel('MSE'); ax.set_title('不同编码器下的 MSE 对比', fontsize=11, weight='bold')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    # 中: 改善幅度
    ax = axes[1]
    imps = [0.5, 3.1, 7.1]
    bars = ax.bar(enc, imps, color=['#FFCDD2', '#FFF9C4', '#C8E6C9'], edgecolor='#78909C', linewidth=1.2)
    for bar, val in zip(bars, imps):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15, f'+{val:.1f}%',
                ha='center', fontsize=13, weight='bold')
    ax.set_ylabel('相比随机基线的改善 (%)'); ax.set_title('核心集改善幅度', fontsize=11, weight='bold')
    ax.set_xticklabels(enc, fontsize=8); ax.axhline(y=0, color='#757575', linestyle='--'); ax.grid(axis='y', alpha=0.3)

    # 右: 热力图
    ax = axes[2]
    td = data.get('temporal', {})
    rand_pd = np.array(td.get('random_per_dim', [0.887, 1.077, 0.924, 0.519, 0.720, 0.943, 1.029]))
    bc_pd = np.array(td.get('braincorest_per_dim', [0.798, 1.055, 0.867, 0.462, 0.623, 0.838, 1.020]))
    imp_t = (rand_pd - bc_pd) / rand_pd * 100

    sc = data.get('single_clip', {})
    if sc:
        rp = np.array(sc.get('random_per_dim', [0.888, 1.102, 0.975, 0.559, 0.773, 0.979, 1.037]))
        bp = np.array(sc.get('braincorest_per_dim', [0.841, 1.061, 0.959, 0.575, 0.698, 0.930, 1.055]))
        imp_s = (rp - bp) / rp * 100
        mat = np.vstack([imp_s, imp_t])
        sns.heatmap(mat, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                    xticklabels=ACTION_CN, yticklabels=['CLIP 单帧', 'CLIP 时序堆叠'],
                    ax=ax, cbar_kws={'label': '改善 %'}, vmin=-5, vmax=15)
    else:
        sns.heatmap(imp_t.reshape(1,-1), annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                    xticklabels=ACTION_CN, yticklabels=['CLIP 时序堆叠'],
                    ax=ax, cbar_kws={'label': '改善 %'}, vmin=-5, vmax=15)
    ax.set_title('各维度改善幅度热力图', fontsize=11, weight='bold')

    fig.suptitle('图 7: BRAIN-Coreset 消融分析', fontsize=14, weight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig7_ablation.png'), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, 'fig7_ablation.pdf'))
    plt.close(); print('图7 已保存')


def main():
    print('生成中文版图表 ...')
    data = load_data()
    fig1(); fig2(); fig3(); fig4(); fig5(data); fig6(data); fig7(data)
    print(f'\n全部 7 张图保存到: {FIG_DIR}')

if __name__ == '__main__':
    main()
