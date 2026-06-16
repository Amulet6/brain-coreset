"""
Baseline 实验：随机采样 10% 训练 MLP，报告 MSE

对应题目要求：
  "基准测试（Baseline）：在指定的 ALOHA 仿真操作数据集上，随机抽取 10% 的轨迹样本。
   利用预训练的轻量级视觉模型（如冻结权重的ResNet-18）离线提取图像特征，
   构建[视觉特征]->[7自由度机械臂动作]的特征回归数据集。
   训练一个轻量级的多层感知机（MLP），并报告动作预测的均方误差（MSE）。"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np
import torch
from src.data_loader import load_aloha_data, get_flat_data
from src.feature_extractor import FeatureExtractor
from src.mlp_model import MLP, train_mlp, evaluate_mlp, set_seed

# 配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_REPEATS = 3  # 随机采样重复次数（取平均）
SEED_BASE = 42

# 动作维度名称（右臂 7-DoF）
ACTION_DIM_NAMES = [
    'right_waist', 'right_shoulder', 'right_elbow',
    'right_forearm_roll', 'right_wrist_angle', 'right_wrist_rotate',
    'right_gripper'
]


def run_baseline():
    """运行完整 Baseline 实验"""
    print('=' * 60)
    print('BRAIN-Coreset Baseline 实验')
    print(f'设备: {DEVICE}')
    print('=' * 60)

    # --- 1. 加载数据 ---
    print('\n[Step 1/5] 加载数据...')
    train_data, test_data, meta = load_aloha_data(use_cache=True)

    # --- 2. 展平 & 标准化 ---
    print('[Step 2/5] 数据预处理...')
    train_frames, train_actions, train_boundaries, a_mean, a_std = get_flat_data(train_data)
    test_frames, test_actions_raw, test_boundaries, _, _ = get_flat_data(test_data, normalize_actions=False)
    # 测试集用训练集统计量标准化
    test_actions = (test_actions_raw - a_mean) / a_std

    print(f'  训练集: {train_frames.shape[0]} 帧, {len(train_boundaries)} episodes')
    print(f'  测试集: {test_frames.shape[0]} 帧, {len(test_boundaries)} episodes')
    print(f'  动作归一化: mean=[{a_mean.flatten()[0]:.3f}, ...], std=[{a_std.flatten()[0]:.3f}, ...]')

    # --- 3. 提取特征 ---
    print('[Step 3/5] 提取 ResNet-18 视觉特征 (with Multi-Crop TTA)...')
    extractor = FeatureExtractor(device=DEVICE)

    train_features = extractor.extract(train_frames, cache_name='train_features', verbose=True)
    test_features = extractor.extract(test_frames, cache_name='test_features', verbose=True)

    print(f'  训练特征: {train_features.shape}')
    print(f'  测试特征: {test_features.shape}')

    # --- 4. Baseline: Random 10% ---
    print(f'\n[Step 4/5] Baseline: Random 10% ({meta["ten_percent_train"]} 帧), {N_REPEATS} 次重复...')

    baseline_results = []
    n_coreset = meta['ten_percent_train']

    for run_idx in range(N_REPEATS):
        seed = SEED_BASE + run_idx
        set_seed(seed)
        np.random.seed(seed)

        # 随机采样 10%
        indices = np.random.choice(train_features.shape[0], size=n_coreset, replace=False)
        subset_features = train_features[indices]
        subset_actions = train_actions[indices]

        # 训练 MLP
        model = MLP()
        history = train_mlp(
            model, subset_features, subset_actions,
            val_features=None, val_actions=None,
            device=DEVICE, verbose=(run_idx == 0)  # 仅第1次显示进度条
        )

        # 评估
        mse, per_dim_mse, predictions = evaluate_mlp(
            model, test_features, test_actions,
            device=DEVICE, return_per_dim=True
        )

        baseline_results.append({
            'run': run_idx,
            'seed': seed,
            'mse': mse,
            'per_dim_mse': per_dim_mse,
            'final_train_loss': history['train_loss'][-1],
        })

        print(f'  Run {run_idx + 1}/{N_REPEATS}: MSE = {mse:.6f}, '
              f'各维度 MSE = [{", ".join(f"{x:.4f}" for x in per_dim_mse)}]')

    # --- 5. 汇总 ---
    print(f'\n[Step 5/5] Baseline 结果汇总:')
    mse_values = [r['mse'] for r in baseline_results]
    mean_mse = np.mean(mse_values)
    std_mse = np.std(mse_values)

    print(f'  MSE (均值 ± 标准差): {mean_mse:.6f} ± {std_mse:.6f}')
    print(f'  各次: {[f"{v:.6f}" for v in mse_values]}')

    # 各维度平均 MSE
    avg_per_dim = np.mean([r['per_dim_mse'] for r in baseline_results], axis=0)
    print(f'  各维度 MSE:')
    for name, val in zip(ACTION_DIM_NAMES, avg_per_dim):
        print(f'    {name:20s}: {val:.6f}')

    # 保存结果
    results = {
        'baseline_mse_mean': float(mean_mse),
        'baseline_mse_std': float(std_mse),
        'baseline_mse_per_run': [float(v) for v in mse_values],
        'baseline_per_dim_mse': avg_per_dim.tolist(),
        'action_dim_names': ACTION_DIM_NAMES,
        'n_repeats': N_REPEATS,
        'n_coreset': n_coreset,
        'total_train_frames': meta['train_frames'],
        'total_test_frames': meta['test_frames'],
        'device': DEVICE,
    }

    np.save('data/cache/baseline_results.npy', results)
    print(f'\n结果已保存到 data/cache/baseline_results.npy')

    print('=' * 60)
    print(f'Baseline MSE: {mean_mse:.6f} ± {std_mse:.6f}')
    print('=' * 60)

    return results


if __name__ == '__main__':
    run_baseline()
