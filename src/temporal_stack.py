"""
时序特征堆叠模块：将单帧特征 v_t 堆叠为 [v_{t-2}, v_{t-1}, v_t]

对应脑机制: 前额叶工作记忆 (Working Memory)
— 大脑不只看当前帧，而是维持一个约 100-200ms 的感觉记忆缓冲，
  将最近几帧的信息整合为"运动趋势"的表征。
— ACT 论文 (Zhao et al. 2023) 的 action chunking 也是类似思想。

用法:
  from src.temporal_stack import stack_features
  stacked = stack_features(features, boundaries)  # (N, 1536)
"""

import numpy as np


def stack_features(features, boundaries, window=3):
    """
    将单帧特征堆叠为时序窗口特征。

    Args:
        features: (N, D) 单帧特征 (如 CLIP 512d)
        boundaries: list of (start, end) 每个 episode 的 [start, end) 范围
        window: 堆叠窗口大小 (默认 3: t-2, t-1, t)

    Returns:
        stacked: (N, D*window) — 保持与原 features 一一对应
                 episode 前 window-1 帧用自身填充 (pad with self)
    """
    N, D = features.shape
    stacked = np.zeros((N, D * window), dtype=np.float32)

    for start, end in boundaries:
        for t in range(start, end):
            # 收集窗口内的特征: [t-window+1, ..., t]
            frames = []
            for offset in range(window - 1, -1, -1):  # 从远到近
                idx = t - offset
                if idx < start:
                    idx = start  # 边界填充: 用首帧
                frames.append(features[idx])

            stacked[t] = np.concatenate(frames)

    return stacked


if __name__ == '__main__':
    # 测试
    features = np.arange(20).reshape(10, 2).astype(np.float32)
    boundaries = [(0, 5), (5, 10)]
    stacked = stack_features(features, boundaries, window=3)
    print(f'Input: {features.shape} -> Stacked: {stacked.shape}')
    print(f'Frame 0 (boundary): {stacked[0]}')   # 应该都是 features[0]
    print(f'Frame 2 (normal):  {stacked[2]}')    # [f0, f1, f2]
    print(f'Frame 5 (new ep):  {stacked[5]}')    # [f5, f5, f5]
    print('OK')
