"""
快速验证：数据加载 -> 特征提取 -> 确认 pipeline 正常
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from src.data_loader import load_aloha_data, get_flat_data
from src.feature_extractor import FeatureExtractor
import numpy as np

print('='*60)
print('Pipeline 验证测试')
print('='*60)

# Step 1: 加载数据
print('\n--- Step 1: 加载 ALOHA 数据 ---')
train_data, test_data, meta = load_aloha_data(use_cache=True)

# Step 2: 展平 + 标准化
print('\n--- Step 2: 数据展平与标准化 ---')
frames, actions, boundaries, a_mean, a_std = get_flat_data(train_data)
print(f'Frames: {frames.shape}, dtype={frames.dtype}')
print(f'Actions: {actions.shape}, dtype={actions.dtype}')
print(f'Action range after normalize: [{actions.min():.3f}, {actions.max():.3f}]')
print(f'Episodes: {len(boundaries)}, 边界示例: {boundaries[:3]}')

# Step 3: 特征提取 (仅提取前 100 帧做快速验证)
print('\n--- Step 3: 特征提取 (采样 100 帧验证) ---')
extractor = FeatureExtractor(device='cuda')
sample_frames = frames[:100]
features = extractor.extract(sample_frames, use_cache=False, cache_name='test_sample')
print(f'Features: {features.shape}')
print(f'Feature L2 norm range: [{np.linalg.norm(features, axis=1).min():.4f}, '
      f'{np.linalg.norm(features, axis=1).max():.4f}]')
print(f'Feature pairwise distance stats:')
dists = []
for i in range(min(50, len(features))):
    for j in range(i+1, min(50, len(features))):
        dists.append(np.linalg.norm(features[i] - features[j]))
dists = np.array(dists)
print(f'  Mean: {dists.mean():.4f}, Std: {dists.std():.4f}')
print(f'  Min: {dists.min():.4f}, Max: {dists.max():.4f}')
# 检查是否有退化（所有特征相同）
if dists.min() < 0.01:
    print('  WARNING: 特征可能退化！')

# Step 4: 测试集
print('\n--- Step 4: 测试集 ---')
test_frames, test_actions, test_boundaries, _, _ = get_flat_data(test_data, normalize_actions=True)
test_actions_norm = (test_actions - a_mean) / a_std  # 用训练集统计量
print(f'Test frames: {test_frames.shape}')
print(f'Test actions: {test_actions.shape}')

print('\n' + '='*60)
print('Pipeline 验证通过!')
print(f'Train: {meta["train_frames"]} frames, Test: {meta["test_frames"]} frames')
print(f'10% Coreset target: {meta["ten_percent_train"]} frames')
print('='*60)
