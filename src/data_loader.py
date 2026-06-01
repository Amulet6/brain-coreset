"""
数据加载模块：ALOHA Sim Transfer Cube (Human Demonstrations)

功能：
  1. 通过 lerobot 加载数据集
  2. 解码 observation.images.top 视频帧
  3. 提取右臂 7-DoF 动作 (indices 7:14)
  4. 按 episode 划分 train/test (40/10)
  5. 缓存帧和动作为本地 .npy 文件

用法：
  from src.data_loader import load_aloha_data
  train_eps, test_eps = load_aloha_data()
"""

import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# 全局配置
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
HF_ENDPOINT = 'https://hf-mirror.com'
REPO_ID = 'lerobot/aloha_sim_transfer_cube_human'
RIGHT_ARM_SLICE = slice(7, 14)  # 右臂 7-DoF: indices 7..13
TRAIN_EPISODES = 40
TEST_EPISODES = 10
TOTAL_EPISODES = 50
IMAGE_SIZE = (480, 640, 3)
FRAME_HEIGHT, FRAME_WIDTH = 224, 224  # ResNet-18 标准输入


def _ensure_cache_dirs():
    """创建缓存目录"""
    os.makedirs(CACHE_DIR, exist_ok=True)


def _get_cache_path(episode_idx, data_type='frames'):
    """获取缓存文件路径"""
    return os.path.join(CACHE_DIR, f'ep_{episode_idx:02d}_{data_type}.npy')


def load_aloha_data(use_cache=True, verbose=True):
    """
    主入口：加载 ALOHA 数据，返回按 episode 组织的数据字典。

    Args:
        use_cache: 是否使用本地缓存（首次运行会自动下载并缓存）
        verbose: 是否打印进度信息

    Returns:
        train_data: dict, key=episode_idx, value={'frames': np.ndarray, 'actions': np.ndarray}
        test_data: dict, 同上
        metadata: dict, 包含数据集统计信息
    """
    _ensure_cache_dirs()

    # 检查缓存
    all_cached = all(
        os.path.exists(_get_cache_path(i, 'frames')) and
        os.path.exists(_get_cache_path(i, 'actions'))
        for i in range(TOTAL_EPISODES)
    )

    if use_cache and all_cached:
        if verbose:
            print('[DataLoader] 从本地缓存加载数据...')
        return _load_from_cache(verbose)

    # 通过 lerobot 下载
    if verbose:
        print('[DataLoader] 首次运行，通过 lerobot 下载数据集 (~200MB)...')
        print(f'[DataLoader] 使用镜像: {HF_ENDPOINT}')

    os.environ['HF_ENDPOINT'] = HF_ENDPOINT

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if verbose:
        print('[DataLoader] 加载数据集...')

    dataset = LeRobotDataset(REPO_ID)

    # 按 episode 组织数据
    if verbose:
        print('[DataLoader] 解码视频帧并按 episode 组织...')
        print(f'[DataLoader] 总帧数: {len(dataset)}, Episode 数: {TOTAL_EPISODES}')

    episodes = defaultdict(lambda: {'frames': [], 'actions': []})

    # 批量处理以减少内存压力
    batch_size = 500
    for start_idx in tqdm(range(0, len(dataset), batch_size),
                          desc='解码视频帧', disable=not verbose):
        end_idx = min(start_idx + batch_size, len(dataset))

        for i in range(start_idx, end_idx):
            sample = dataset[i]
            ep_idx = sample['episode_index'].item() if hasattr(sample['episode_index'], 'item') else sample['episode_index']

            # 解码视频帧: observation.images.top 是 (H, W, 3) numpy array
            frame = sample['observation.images.top']
            if hasattr(frame, 'numpy'):
                frame = frame.numpy()
            frame = np.asarray(frame, dtype=np.uint8)

            # 右臂动作: indices 7:14
            action = np.array(sample['action'][RIGHT_ARM_SLICE], dtype=np.float32)

            episodes[ep_idx]['frames'].append(frame)
            episodes[ep_idx]['actions'].append(action)

    # 转为 numpy 数组并缓存
    if verbose:
        print('[DataLoader] 缓存数据到本地...')

    for ep_idx in range(TOTAL_EPISODES):
        frames = np.stack(episodes[ep_idx]['frames'], axis=0)  # (T, H, W, 3)
        actions = np.stack(episodes[ep_idx]['actions'], axis=0)  # (T, 7)

        np.save(_get_cache_path(ep_idx, 'frames'), frames)
        np.save(_get_cache_path(ep_idx, 'actions'), actions)
        episodes[ep_idx] = {'frames': frames, 'actions': actions}

    if verbose:
        print('[DataLoader] 缓存完成。')

    return _split_and_return(episodes, verbose)


def _load_from_cache(verbose=True):
    """从本地缓存加载数据"""
    episodes = {}
    for i in tqdm(range(TOTAL_EPISODES), desc='加载缓存', disable=not verbose):
        frames = np.load(_get_cache_path(i, 'frames'))
        actions = np.load(_get_cache_path(i, 'actions'))
        episodes[i] = {'frames': frames, 'actions': actions}
    return _split_and_return(episodes, verbose)


def _split_and_return(episodes, verbose=True):
    """划分 train/test 并返回"""
    # 固定划分: 前 40 个 episode 训练, 后 10 个测试
    train_data = {i: episodes[i] for i in range(TRAIN_EPISODES)}
    test_data = {i: episodes[i] for i in range(TRAIN_EPISODES, TOTAL_EPISODES)}

    # 统计信息
    total_train_frames = sum(v['frames'].shape[0] for v in train_data.values())
    total_test_frames = sum(v['frames'].shape[0] for v in test_data.values())
    total_frames = total_train_frames + total_test_frames

    metadata = {
        'total_frames': total_frames,
        'train_frames': total_train_frames,
        'test_frames': total_test_frames,
        'train_episodes': TRAIN_EPISODES,
        'test_episodes': TEST_EPISODES,
        'action_dim': 7,
        'image_shape': IMAGE_SIZE,
        'ten_percent_train': int(total_train_frames * 0.1),
        'ten_percent_total': int(total_frames * 0.1),
    }

    if verbose:
        print(f'[DataLoader] 数据加载完成:')
        print(f'  训练集: {TRAIN_EPISODES} episodes, {total_train_frames} 帧')
        print(f'  测试集: {TEST_EPISODES} episodes, {total_test_frames} 帧')
        print(f'  10% 核心集目标: {metadata["ten_percent_train"]} 帧 (训练集)')
        print(f'  动作维度: 7 (右臂)')

    return train_data, test_data, metadata


def get_flat_data(episodes_dict, normalize_actions=True):
    """
    将按 episode 组织的字典展平为连续数组。

    Args:
        episodes_dict: {ep_idx: {'frames': (T,H,W,3), 'actions': (T,7)}}
        normalize_actions: 是否对动作做 Z-score 标准化

    Returns:
        all_frames: (N, H, W, 3)
        all_actions: (N, 7)
        episode_boundaries: list of (start, end) 每个 episode 在展平数组中的范围
        action_mean, action_std: 标准化参数 (if normalize_actions else None)
    """
    frame_list, action_list, boundaries = [], [], []
    offset = 0

    for ep_idx in sorted(episodes_dict.keys()):
        ep = episodes_dict[ep_idx]
        n = ep['frames'].shape[0]
        frame_list.append(ep['frames'])
        action_list.append(ep['actions'])
        boundaries.append((offset, offset + n))
        offset += n

    all_frames = np.concatenate(frame_list, axis=0)
    all_actions = np.concatenate(action_list, axis=0)

    action_mean, action_std = None, None
    if normalize_actions:
        action_mean = all_actions.mean(axis=0, keepdims=True)
        action_std = all_actions.std(axis=0, keepdims=True)
        action_std = np.where(action_std < 1e-8, 1.0, action_std)  # 防止除零
        all_actions = (all_actions - action_mean) / action_std

    return all_frames, all_actions, boundaries, action_mean, action_std


if __name__ == '__main__':
    # 测试：加载数据
    train_data, test_data, meta = load_aloha_data(use_cache=True)
    frames, actions, boundaries, mean, std = get_flat_data(train_data)
    print(f'\n展平数据: frames={frames.shape}, actions={actions.shape}')
    print(f'动作均值: {mean.flatten()}')
    print(f'动作标准差: {std.flatten()}')
    print(f'Episode 边界 (前5): {boundaries[:5]}')
