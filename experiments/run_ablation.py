"""
消融实验: 一键复现 v1 → v5 全部版本

v1: ResNet-18 + 高误差采样 + RAS + FL (512d, 单头MLP)
v2: ResNet-18 + 原型采样  + RAS + FL (512d, 单头MLP)
v3: CLIP 单帧   + 原型采样  + RAS + FL (512d, 单头MLP)
v4: CLIP 时序   + 原型采样  + RAS + FL (1536d, 单头MLP)
v5: CLIP VLA    + 原型采样  + RAS + FL (2048d, 双头MLP)

用法: python experiments/run_ablation.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np; np.random.seed(42)
import torch
from src.data_loader import load_aloha_data, get_flat_data
from src.feature_extractor import FeatureExtractor, get_language_embedding
from src.temporal_stack import stack_features, concat_language
from src.mlp_model import (MLP, DualHeadMLP, train_mlp, evaluate_mlp,
                           train_dual_mlp, evaluate_dual_mlp, set_seed)
from src.stage1_predictive_coding import predictive_coding_filter
from src.stage2_ras_events import ras_event_detection
from src.stage3_facility_location import facility_location_selection

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_TRIALS = 3
SEED_BASE = 42
N_CORESET = 1600  # 10% of 16000
ACTION_NAMES = ['Waist','Shoulder','Elbow','F.Roll','W.Angle','W.Rot','Gripper']


def get_coreset_indices(features, actions, boundaries, keep_high_error, n_coreset):
    """通用核心集选择: 阶段1(指定模式) + 阶段2 + 阶段3"""
    s1_mask, s1_info = predictive_coding_filter(
        actions, boundaries, target_ratio=0.35,
        keep_high_error=keep_high_error, verbose=False
    )
    s2_mask, s2_info = ras_event_detection(actions, boundaries, verbose=False)
    cand_mask = s1_mask | s2_mask
    coreset_idx, s3_info = facility_location_selection(
        features, actions, cand_mask, n_coreset, verbose=False
    )
    return coreset_idx, {
        's1_retention': float(s1_mask.mean()),
        's2_retention': float(s2_mask.mean()),
        'cand_ratio': float(cand_mask.mean()),
        'n_coreset': len(coreset_idx),
    }


def run_single_head(name, tr_feat, te_feat, tr_a, te_a, coreset_idx, n_coreset):
    """v1-v4: 单头 MLP, 7-DoF MSE"""
    rand_mses, core_mses = [], []
    for run in range(N_TRIALS):
        np.random.seed(100 + run); set_seed(100 + run)
        ridx = np.random.choice(len(tr_feat), size=n_coreset, replace=False)
        m = MLP(input_dim=tr_feat.shape[1])
        train_mlp(m, tr_feat[ridx], tr_a[ridx], device=DEVICE, verbose=False)
        rand_mses.append(evaluate_mlp(m, te_feat, te_a, device=DEVICE))

        set_seed(SEED_BASE + run)
        m2 = MLP(input_dim=tr_feat.shape[1])
        train_mlp(m2, tr_feat[coreset_idx], tr_a[coreset_idx], device=DEVICE, verbose=False)
        core_mses.append(evaluate_mlp(m2, te_feat, te_a, device=DEVICE))

    r_mean, r_std = np.mean(rand_mses), np.std(rand_mses)
    c_mean, c_std = np.mean(core_mses), np.std(core_mses)
    imp = (r_mean - c_mean) / r_mean * 100
    return {'name': name, 'random_mse': float(r_mean), 'random_std': float(r_std),
            'coreset_mse': float(c_mean), 'coreset_std': float(c_std),
            'improvement_pct': float(imp)}


def run_dual_head(name, tr_feat, te_feat, tr_a, te_a, coreset_idx, n_coreset):
    """v5: 双头 MLP, 6-DoF MSE + 夹爪 Acc/F1"""
    rand_m6, rand_acc, rand_f1 = [], [], []
    core_m6, core_acc, core_f1 = [], [], []
    for run in range(N_TRIALS):
        np.random.seed(100 + run); set_seed(100 + run)
        ridx = np.random.choice(len(tr_feat), size=n_coreset, replace=False)
        m = DualHeadMLP(input_dim=tr_feat.shape[1])
        train_dual_mlp(m, tr_feat[ridx], tr_a[ridx], device=DEVICE, verbose=False)
        m6, a, f, _ = evaluate_dual_mlp(m, te_feat, te_a, device=DEVICE)
        rand_m6.append(m6); rand_acc.append(a); rand_f1.append(f)

        set_seed(SEED_BASE + run)
        m2 = DualHeadMLP(input_dim=tr_feat.shape[1])
        train_dual_mlp(m2, tr_feat[coreset_idx], tr_a[coreset_idx], device=DEVICE, verbose=False)
        m6_c, a_c, f_c, _ = evaluate_dual_mlp(m2, te_feat, te_a, device=DEVICE)
        core_m6.append(m6_c); core_acc.append(a_c); core_f1.append(f_c)

    r_m = np.mean(rand_m6); c_m = np.mean(core_m6)
    imp = (r_m - c_m) / r_m * 100
    return {'name': name,
            'random_mse6': float(r_m), 'coreset_mse6': float(c_m),
            'improvement_mse6': float(imp),
            'random_acc': float(np.mean(rand_acc)), 'coreset_acc': float(np.mean(core_acc)),
            'random_f1': float(np.mean(rand_f1)), 'coreset_f1': float(np.mean(core_f1))}


def main():
    print('=' * 60)
    print('BRAIN-Coreset 消融实验: v1 → v5')
    print('=' * 60)

    # Load data (shared across all versions)
    train_data, test_data, meta = load_aloha_data(use_cache=True)
    tr_frames, tr_actions, tr_bounds, a_mean, a_std = get_flat_data(train_data)
    te_frames, te_actions_raw, te_bounds, _, _ = get_flat_data(test_data, normalize_actions=False)
    te_actions = (te_actions_raw - a_mean) / a_std
    n_coreset = meta['ten_percent_train']

    results = {}

    # ========================
    # v1: ResNet-18 + 高误差
    # ========================
    print('\n--- v1: ResNet-18 + 高误差采样 ---')
    ext_rn = FeatureExtractor(device=DEVICE, encoder_type='resnet')
    tr_f1 = ext_rn.extract(tr_frames, cache_name='train_features', use_cache=True)
    te_f1 = ext_rn.extract(te_frames, cache_name='test_features', use_cache=True)
    coreset_v1, info_v1 = get_coreset_indices(tr_f1, tr_actions, tr_bounds, keep_high_error=True, n_coreset=n_coreset)
    results['v1'] = run_single_head('v1 ResNet 高误差', tr_f1, te_f1, tr_actions, te_actions, coreset_v1, n_coreset)
    print(f"  Random: {results['v1']['random_mse']:.4f}, Coreset: {results['v1']['coreset_mse']:.4f} ({results['v1']['improvement_pct']:+.1f}%)")

    # ========================
    # v2: ResNet-18 + 原型
    # ========================
    print('\n--- v2: ResNet-18 + 原型采样 ---')
    coreset_v2, info_v2 = get_coreset_indices(tr_f1, tr_actions, tr_bounds, keep_high_error=False, n_coreset=n_coreset)
    results['v2'] = run_single_head('v2 ResNet 原型', tr_f1, te_f1, tr_actions, te_actions, coreset_v2, n_coreset)
    print(f"  Random: {results['v2']['random_mse']:.4f}, Coreset: {results['v2']['coreset_mse']:.4f} ({results['v2']['improvement_pct']:+.1f}%)")

    # ========================
    # v3: CLIP 单帧 + 原型
    # ========================
    print('\n--- v3: CLIP 单帧 + 原型采样 ---')
    ext = FeatureExtractor(device=DEVICE, encoder_type='clip')
    tr_feat_raw = ext.extract(tr_frames, cache_name='train_features_clip', use_cache=True)
    te_feat_raw = ext.extract(te_frames, cache_name='test_features_clip', use_cache=True)
    coreset_v3, info_v3 = get_coreset_indices(tr_feat_raw, tr_actions, tr_bounds, keep_high_error=False, n_coreset=n_coreset)
    results['v3'] = run_single_head('v3 CLIP 单帧', tr_feat_raw, te_feat_raw, tr_actions, te_actions, coreset_v3, n_coreset)
    print(f"  Random: {results['v3']['random_mse']:.4f}, Coreset: {results['v3']['coreset_mse']:.4f} ({results['v3']['improvement_pct']:+.1f}%)")

    # ========================
    # v4: CLIP 时序 + 原型
    # ========================
    print('\n--- v4: CLIP 时序堆叠 + 原型采样 ---')
    tr_feat_v4 = stack_features(tr_feat_raw, tr_bounds, window=3)
    te_feat_v4 = stack_features(te_feat_raw, te_bounds, window=3)
    coreset_v4, info_v4 = get_coreset_indices(tr_feat_raw, tr_actions, tr_bounds, keep_high_error=False, n_coreset=n_coreset)
    results['v4'] = run_single_head('v4 CLIP 时序', tr_feat_v4, te_feat_v4, tr_actions, te_actions, coreset_v4, n_coreset)
    print(f"  Random: {results['v4']['random_mse']:.4f}, Coreset: {results['v4']['coreset_mse']:.4f} ({results['v4']['improvement_pct']:+.1f}%)")

    # ========================
    # v5: CLIP VLA + 双头 MLP
    # ========================
    print('\n--- v5: CLIP VLA + 双头 MLP ---')
    lang_emb = get_language_embedding(device=DEVICE)
    tr_feat_v5 = concat_language(tr_feat_v4, lang_emb)
    te_feat_v5 = concat_language(te_feat_v4, lang_emb)
    coreset_v5, info_v5 = get_coreset_indices(tr_feat_raw, tr_actions, tr_bounds, keep_high_error=False, n_coreset=n_coreset)
    results['v5'] = run_dual_head('v5 CLIP VLA', tr_feat_v5, te_feat_v5, tr_actions, te_actions, coreset_v5, n_coreset)
    print(f"  Random MSE_6d: {results['v5']['random_mse6']:.4f}, Coreset MSE_6d: {results['v5']['coreset_mse6']:.4f} ({results['v5']['improvement_mse6']:+.1f}%)")
    print(f"  Random F1: {results['v5']['random_f1']:.3f}, Coreset F1: {results['v5']['coreset_f1']:.3f}")

    # ========================
    # Summary
    # ========================
    print('\n' + '=' * 60)
    print('消融实验汇总')
    print('=' * 60)
    print(f"{'版本':15s} {'输入':>8s} {'策略':15s} {'Random':>10s} {'Coreset':>10s} {'改善':>8s}")
    print('-' * 65)
    for v, r in results.items():
        if 'coreset_mse' in r:
            print(f"{r['name']:15s} {v:>8s} {'—':15s} {r['random_mse']:10.4f} {r['coreset_mse']:10.4f} {r['improvement_pct']:+7.1f}%")
        else:
            print(f"{r['name']:15s} {v:>8s} {'—':15s} {r['random_mse6']:10.4f} {r['coreset_mse6']:10.4f} {r['improvement_mse6']:+7.1f}% (MSE_6d)")
            print(f"{'':15s} {'':>8s} {'夹爪 F1':>15s} {r['random_f1']:10.3f} {r['coreset_f1']:10.3f}")

    np.save('data/cache/ablation_results.npy', results)
    print(f"\n结果已保存: data/cache/ablation_results.npy")


if __name__ == '__main__':
    main()
