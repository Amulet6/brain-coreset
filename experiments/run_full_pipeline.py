"""
BRAIN-Coreset 完整流水线实验

流程:
  1. 加载数据 + 特征
  2. Stage 1: 预测编码时序过滤
  3. Stage 2: RAS 关键事件检测
  4. 候选池 = Stage1 ∪ Stage2 (目标 25%)
  5. 二分搜索 k 值精确控制候选池大小
  6. Stage 3: Facility Location 子模优化 → 10% 核心集
  7. MLP 训练 + MSE 评估
  8. 与 Baseline 对比
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
from src.stage1_predictive_coding import predictive_coding_filter
from src.stage2_ras_events import ras_event_detection
from src.stage3_facility_location import facility_location_selection

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TARGET_CANDIDATE_RATIO = 0.25  # 候选池目标 25%
N_REPEATS = 3
SEED_BASE = 42


def main():
    print('=' * 60)
    print('BRAIN-Coreset 完整流水线')
    print(f'设备: {DEVICE}')
    print('=' * 60)

    # --- 1. 加载数据 ---
    print('\n[1/7] 加载数据与特征...')
    train_data, test_data, meta = load_aloha_data(use_cache=True)
    train_frames, train_actions, train_boundaries, a_mean, a_std = get_flat_data(train_data)
    test_frames, test_actions_raw, test_boundaries, _, _ = get_flat_data(test_data, normalize_actions=False)
    test_actions = (test_actions_raw - a_mean) / a_std

    extractor = FeatureExtractor(device=DEVICE)
    train_features = extractor.extract(train_frames, cache_name='train_features')
    test_features = extractor.extract(test_frames, cache_name='test_features')

    n_coreset = meta['ten_percent_train']
    print(f'训练帧: {train_features.shape}, 测试帧: {test_features.shape}')
    print(f'10% 核心集目标: {n_coreset} 帧')

    # --- 2. Stage 1: 预测编码 (原型采样模式, keep_high_error=False) ---
    print('\n[2/7] Stage 1: 预测编码时序过滤 (原型采样: 保留低误差帧)...')
    # v2: keep_high_error=False → 保留低预测误差的典型帧
    # v1 (已废弃): keep_high_error=True → 保留高误差帧, 实验结果表明更差
    s1_mask_init, s1_info = predictive_coding_filter(
        train_actions, train_boundaries, target_ratio=0.35, keep_high_error=False, verbose=True
    )

    # --- 3. Stage 2: RAS 事件 ---
    print('\n[3/7] Stage 2: RAS 关键事件检测...')
    s2_mask, s2_info = ras_event_detection(train_actions, train_boundaries, verbose=True)

    # --- 4. 检查并调整候选池大小 ---
    print(f'\n[4/7] 候选池分析...')
    candidate_mask = s1_mask_init | s2_mask
    candidate_ratio = candidate_mask.mean()

    print(f'  Stage 1 (原型采样): {s1_mask_init.sum()} 帧 ({s1_mask_init.mean()*100:.1f}%)')
    print(f'  Stage 2 (RAS事件):   {s2_mask.sum()} 帧 ({s2_mask.mean()*100:.1f}%)')
    print(f'  候选池 (并集):      {candidate_mask.sum()} 帧 ({candidate_ratio*100:.1f}%)')
    print(f'  交集:               {(s1_mask_init & s2_mask).sum()} 帧')

    # 如果候选池太小(<15%)或太大(>40%), 调整 Stage 1 目标比例
    if candidate_ratio < 0.15:
        print(f'  候选池过小, 放宽 Stage 1 至 target_ratio=0.50...')
        s1_mask, _ = predictive_coding_filter(
            train_actions, train_boundaries, target_ratio=0.50,
            keep_high_error=False, verbose=False
        )
        candidate_mask = s1_mask | s2_mask
        print(f'  调整后候选池: {candidate_mask.mean()*100:.1f}%')
    elif candidate_ratio > 0.40:
        print(f'  候选池过大, 收紧 Stage 1 至 target_ratio=0.25...')
        s1_mask, _ = predictive_coding_filter(
            train_actions, train_boundaries, target_ratio=0.25,
            keep_high_error=False, verbose=False
        )
        candidate_mask = s1_mask | s2_mask
        print(f'  调整后候选池: {candidate_mask.mean()*100:.1f}%')
    else:
        s1_mask = s1_mask_init
        print(f'  候选池大小合适, 直接进入 Stage 3')

    # --- 5. Stage 3: Facility Location ---
    print(f'\n[5/7] Stage 3: Facility Location → {n_coreset} 帧核心集...')
    coreset_indices, s3_info = facility_location_selection(
        train_features, train_actions, candidate_mask, n_coreset, verbose=True
    )

    # --- 6. MLP 训练 ---
    print(f'\n[6/7] 核心集 MLP 训练 ({N_REPEATS} 次取平均)...')
    coreset_results = []

    for run_idx in range(N_REPEATS):
        set_seed(SEED_BASE + run_idx)
        subset_features = train_features[coreset_indices]
        subset_actions = train_actions[coreset_indices]

        model = MLP()
        history = train_mlp(
            model, subset_features, subset_actions,
            device=DEVICE, verbose=(run_idx == 0)
        )

        mse, per_dim_mse, _ = evaluate_mlp(
            model, test_features, test_actions,
            device=DEVICE, return_per_dim=True
        )

        coreset_results.append({
            'run': run_idx, 'mse': mse, 'per_dim_mse': per_dim_mse
        })
        print(f'  Run {run_idx+1}: MSE = {mse:.6f}')

    # --- 7. 结果对比 ---
    print(f'\n[7/7] 结果对比:')
    mse_values = [r['mse'] for r in coreset_results]
    mean_mse = np.mean(mse_values)
    std_mse = np.std(mse_values)

    # 加载 baseline
    baseline = np.load('data/cache/baseline_results.npy', allow_pickle=True).item()
    baseline_mse = baseline['baseline_mse_mean']

    improvement = (baseline_mse - mean_mse) / baseline_mse * 100

    print(f'  Baseline (Random 10%): MSE = {baseline_mse:.6f}')
    print(f'  BRAIN-Coreset (10%):   MSE = {mean_mse:.6f} ± {std_mse:.6f}')
    print(f'  改善幅度: {improvement:+.1f}%')

    avg_per_dim = np.mean([r['per_dim_mse'] for r in coreset_results], axis=0)
    baseline_per_dim = np.array(baseline['baseline_per_dim_mse'])

    print(f'\n  各维度对比:')
    for i, name in enumerate(baseline['action_dim_names']):
        delta = (baseline_per_dim[i] - avg_per_dim[i]) / baseline_per_dim[i] * 100
        print(f'    {name:20s}: {baseline_per_dim[i]:.4f} → {avg_per_dim[i]:.4f} ({delta:+.1f}%)')

    # 保存
    results = {
        'coreset_mse_mean': float(mean_mse),
        'coreset_mse_std': float(std_mse),
        'baseline_mse': float(baseline_mse),
        'improvement_pct': float(improvement),
        'coreset_per_dim_mse': avg_per_dim.tolist(),
        'baseline_per_dim_mse': baseline_per_dim.tolist(),
        'n_coreset': n_coreset,
        'candidate_ratio': float(candidate_mask.mean()),
        'stage1_retention': float(s1_mask.mean()),
        'stage2_retention': float(s2_mask.mean()),
        'stage3_info': s3_info,
    }
    np.save('data/cache/braincorest_results.npy', results)

    print(f'\n结果已保存到 data/cache/braincorest_results.npy')
    print('=' * 60)
    print(f'BRAIN-Coreset MSE: {mean_mse:.6f} (vs Baseline {baseline_mse:.6f}, {improvement:+.1f}%)')
    print('=' * 60)

    return results


if __name__ == '__main__':
    main()
