"""深入探索 ALOHA 数据集 — 检查图像数据"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset, get_dataset_config_names
import numpy as np

# 1. 所有 configs
print('=== Available Configs ===')
try:
    configs = get_dataset_config_names('lerobot/aloha_sim_transfer_cube_human')
    print(f'Configs: {configs}')
except Exception as e:
    print(f'Error: {e}')

# 2. 尝试不同 config 加载
for cfg in (configs if 'configs' in dir() else [None, 'default', 'all']):
    try:
        if cfg is None:
            ds = load_dataset('lerobot/aloha_sim_transfer_cube_human', split='train')
        else:
            ds = load_dataset('lerobot/aloha_sim_transfer_cube_human', cfg, split='train')
        print(f'\nConfig="{cfg}": {len(ds)} entries, features={list(ds.features.keys())}')
        # Check for image fields
        for k in ds.features:
            feat = ds.features[k]
            if 'image' in str(feat).lower() or 'Image' in str(type(feat).__name__):
                print(f'  IMAGE FIELD: {k} -> {feat}')
    except Exception as e:
        print(f'Config="{cfg}": ERROR - {str(e)[:200]}')

# 3. 直接查看原始数据文件
print('\n=== Raw Files on HuggingFace ===')
from huggingface_hub import list_repo_files
try:
    files = list_repo_files('lerobot/aloha_sim_transfer_cube_human')
    print(f'Total files: {len(files)}')
    # Show first 30
    for f in files[:30]:
        print(f'  {f}')
    if len(files) > 30:
        print(f'  ... and {len(files)-30} more')
except Exception as e:
    print(f'Error listing files: {e}')

# 4. 用 lerobot 库加载
print('\n=== Try lerobot library ===')
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds_lerobot = LeRobotDataset('lerobot/aloha_sim_transfer_cube_human')
    print(f'LeRobotDataset loaded: {len(ds_lerobot)} frames')
    print(f'Features: {ds_lerobot.features}')
    # Check for image keys
    for k in ds_lerobot.hf_dataset.features:
        print(f'  {k}: {ds_lerobot.hf_dataset.features[k]}')
except Exception as e:
    print(f'Error with LeRobotDataset: {type(e).__name__}: {str(e)[:300]}')

# 5. 检查本地缓存
print('\n=== Local Cache ===')
cache_dir = os.path.expanduser('~/.cache/huggingface/hub/datasets--lerobot--aloha_sim_transfer_cube_human')
if os.path.exists(cache_dir):
    for root, dirs, files in os.walk(cache_dir):
        level = root.replace(cache_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:
            size_mb = os.path.getsize(os.path.join(root, file)) / 1e6
            print(f'{subindent}{file} ({size_mb:.1f} MB)')
        if len(files) > 5:
            print(f'{subindent}... and {len(files)-5} more files')
        if level > 2:
            break
else:
    print('No cache found at default location')

print('\nDone.')
