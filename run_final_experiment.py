"""最终实验: VLA语言模态 + 夹爪分类解耦 + 全候选池"""
import sys, os; sys.path.insert(0, '.')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import numpy as np; np.random.seed(42)
import torch
from src.data_loader import load_aloha_data, get_flat_data
from src.feature_extractor import FeatureExtractor, get_language_embedding
from src.temporal_stack import stack_features, concat_language
from src.mlp_model import DualHeadMLP, train_dual_mlp, evaluate_dual_mlp, set_seed
from src.stage1_predictive_coding import predictive_coding_filter
from src.stage2_ras_events import ras_event_detection
from src.stage3_facility_location import facility_location_selection

DEVICE = 'cuda'; N_TRIALS = 3

train_data, test_data, meta = load_aloha_data(use_cache=True)
tr_f, tr_a, tr_b, am, astd = get_flat_data(train_data)
te_f, te_a_r, te_b, _, _ = get_flat_data(test_data, normalize_actions=False)
te_a = (te_a_r - am) / astd

print('Language embedding...')
lang_emb = get_language_embedding(device=DEVICE)
print(f'  dim={lang_emb.shape[0]}, norm={np.linalg.norm(lang_emb):.3f}')

ext = FeatureExtractor(device=DEVICE, encoder_type='clip')
tr_feat_raw = ext.extract(tr_f, cache_name='train_features_clip', use_cache=True)
te_feat_raw = ext.extract(te_f, cache_name='test_features_clip', use_cache=True)

tr_feat_v = stack_features(tr_feat_raw, tr_b, window=3)
te_feat_v = stack_features(te_feat_raw, te_b, window=3)
tr_feat = concat_language(tr_feat_v, lang_emb)
te_feat = concat_language(te_feat_v, lang_emb)
print(f'Features: train={tr_feat.shape}, test={te_feat.shape} (3x512 visual + 512 lang = 2048)')

n_total = tr_feat.shape[0]; n_10pct = meta['ten_percent_train']

# Full 100%
print('\n=== Full 100% ===')
m = DualHeadMLP(input_dim=2048); set_seed(42)
train_dual_mlp(m, tr_feat, tr_a, device=DEVICE, verbose=False)
mse6_full, acc_full, f1_full, _ = evaluate_dual_mlp(m, te_feat, te_a, device=DEVICE)
print(f'MSE_6d={mse6_full:.4f}, Gripper Acc={acc_full:.2%}, F1={f1_full:.3f}')

# Random 10%
print(f'\n=== Random 10% x{N_TRIALS} ===')
rand_m6, rand_acc, rand_f1 = [], [], []
for run in range(N_TRIALS):
    np.random.seed(100+run); set_seed(100+run)
    ridx = np.random.choice(n_total, size=n_10pct, replace=False)
    m = DualHeadMLP(input_dim=2048)
    train_dual_mlp(m, tr_feat[ridx], tr_a[ridx], device=DEVICE, verbose=False)
    m6, a, f, _ = evaluate_dual_mlp(m, te_feat, te_a, device=DEVICE)
    rand_m6.append(m6); rand_acc.append(a); rand_f1.append(f)
r_m6, r_a, r_f = np.mean(rand_m6), np.mean(rand_acc), np.mean(rand_f1)
print(f'MSE_6d={r_m6:.4f}+/-{np.std(rand_m6):.4f}, Acc={r_a:.2%}, F1={r_f:.3f}')

# BRAIN-Coreset
print('\n=== BRAIN-Coreset (full candidate pool) ===')
s1, _ = predictive_coding_filter(tr_a, tr_b, target_ratio=0.35, keep_high_error=False, verbose=False)
s2, _ = ras_event_detection(tr_a, tr_b, verbose=False)
cand = s1 | s2
print(f'Candidate pool: {cand.sum()} frames ({cand.mean()*100:.1f}%)')

bc_idx, s3_info = facility_location_selection(tr_feat_raw, tr_a, cand, n_10pct, verbose=True)

bc_m6, bc_acc, bc_f1 = [], [], []
for run in range(N_TRIALS):
    set_seed(42+run)
    m = DualHeadMLP(input_dim=2048)
    train_dual_mlp(m, tr_feat[bc_idx], tr_a[bc_idx], device=DEVICE, verbose=False)
    m6, a, f, _ = evaluate_dual_mlp(m, te_feat, te_a, device=DEVICE)
    bc_m6.append(m6); bc_acc.append(a); bc_f1.append(f)
bc_m6_avg, bc_a_avg, bc_f_avg = np.mean(bc_m6), np.mean(bc_acc), np.mean(bc_f1)

print(f'\n====== FINAL VLA RESULTS ======')
print(f'{"Method":20s}  {"MSE_6d":>8s}  {"Gripper Acc":>12s}  {"Gripper F1":>10s}')
print(f'{"Full 100%":20s}  {mse6_full:8.4f}  {acc_full:12.2%}  {f1_full:10.3f}')
print(f'{"Random 10%":20s}  {r_m6:8.4f}  {r_a:12.2%}  {r_f:10.3f}')
print(f'{"BRAIN-Coreset 10%":20s}  {bc_m6_avg:8.4f}  {bc_a_avg:12.2%}  {bc_f_avg:10.3f}')
imp6 = (r_m6 - bc_m6_avg) / r_m6 * 100
print(f'\nMSE_6d improvement: {imp6:+.1f}%')
print(f'Gripper Acc delta: {bc_a_avg - r_a:+.2%}')
print(f'Gripper F1 delta:  {bc_f_avg - r_f:+.3f}')

# Save
results = {
    'full_mse6': float(mse6_full), 'full_acc': float(acc_full), 'full_f1': float(f1_full),
    'random_mse6': float(r_m6), 'random_acc': float(r_a), 'random_f1': float(r_f),
    'coreset_mse6': float(bc_m6_avg), 'coreset_acc': float(bc_a_avg), 'coreset_f1': float(bc_f_avg),
    'mse6_improvement': float(imp6),
    'input_dim': 2048, 'encoder': 'CLIP ViT-B-32 visual+text, temporal x3',
    'n_trials': N_TRIALS, 'n_coreset': n_10pct, 'candidate_size': int(cand.sum()),
}
np.save('data/cache/final_vla_results.npy', results)
print('\nSaved to data/cache/final_vla_results.npy')
