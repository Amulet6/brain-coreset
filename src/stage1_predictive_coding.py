"""
Stage 1: 预测编码时序过滤 (Predictive Coding Temporal Filter)

对应脑机制: Predictive Coding (Millidge et al., 2021)
— 大脑持续生成预测，只在预测误差打破预期时激活神经元

算法:
  对每个 episode 独立处理:
    1. 用全局线性模型 W (R^{21} → R^7) 拟合 a_{t-s-3:t-s-1} → a_{t-s}
       (训练样本与目标帧严格隔离，至少隔 1 帧)
    2. 对每个 t ∈ [3, T-1]:
       x_t = flatten([a_{t-3}, a_{t-2}, a_{t-1}])  # 21d
       â_t = x_t @ W
       e_t = ||a_t - â_t||_2
    3. 标记 e_t > μ + k·σ 的帧 (k 通过二分搜索使输出≈35-40%)

用法:
  from src.stage1_predictive_coding import predictive_coding_filter
  selected_indices, info = predictive_coding_filter(actions, boundaries)
"""

import numpy as np
from collections import defaultdict


def _build_linear_model(actions, exclude_idx):
    """
    构建全局线性模型 W: R^{21} → R^7

    使用除 exclude_idx 及其相邻帧外的所有帧作为训练样本。
    因果约束: 只用 t-3:t-1 的历史帧预测 t。

    Args:
        actions: (T, 7) 动作序列
        exclude_idx: 要排除的帧索引（目标帧，防止信息泄漏）

    Returns:
        W: (21, 7) 线性变换矩阵
        intercept: (7,) 截距项（可选，提升拟合质量）
    """
    T = actions.shape[0]
    X_list, y_list = [], []

    for t in range(3, T):
        # 跳过目标帧附近的样本（防止因果泄漏）
        if abs(t - exclude_idx) <= 1:
            continue
        if abs(t - 3 - exclude_idx) <= 1:
            continue

        x = actions[t-3:t].flatten()  # (21,)
        y = actions[t]                 # (7,)
        X_list.append(x)
        y_list.append(y)

    if len(X_list) < 10:
        # 样本太少，返回零矩阵
        return np.zeros((21, 7)), np.zeros(7)

    X = np.stack(X_list, axis=0)  # (N, 21)
    y = np.stack(y_list, axis=0)  # (N, 7)

    # 加入截距项
    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])  # (N, 22)

    # 最小二乘解
    try:
        W_aug = np.linalg.lstsq(X_aug, y, rcond=None)[0]  # (22, 7)
        W = W_aug[:21, :]       # (21, 7)
        intercept = W_aug[21, :] # (7,)
    except np.linalg.LinAlgError:
        W = np.zeros((21, 7))
        intercept = np.zeros(7)

    return W, intercept


def _compute_prediction_errors(actions, W, intercept):
    """
    逐帧计算预测误差。

    Args:
        actions: (T, 7)
        W: (21, 7)
        intercept: (7,)

    Returns:
        errors: (T-3,) 每个 t≥3 的预测误差（前 3 帧无预测，设为 inf）
    """
    T = actions.shape[0]
    errors = np.full(T, np.inf)

    for t in range(3, T):
        x = actions[t-3:t].flatten()  # (21,)
        pred = x @ W + intercept       # (7,)
        errors[t] = np.linalg.norm(actions[t] - pred)

    return errors


def predictive_coding_filter(actions, boundaries, target_ratio=0.375,
                           keep_high_error=False, verbose=True):
    """
    阶段1: 预测编码时序过滤

    Args:
        actions: (N_total, 7) 全部训练帧的动作（未归一化）
        boundaries: list of (start, end) 每个 episode 的边界
        target_ratio: 目标保留比例 (默认 37.5%)
        keep_high_error:
            True  = 保留高误差帧 (v1 设计，对应"意外打破预测→神经元激活"的逻辑)
                    结果：MSE 7.11，比随机 Baseline 6.71 差 6%。
                    原因：高误差帧=离群点，数据修剪文献(Sorscher et al. 2022)指出
                    原型样本(低误差帧)比困难样本(高误差帧)更有助于泛化。
            False = 保留低误差帧 (v2 修正，对应"高效编码→内部模型表征"的逻辑)
                    保留轨迹中平滑、可预测的典型帧作为原型样本。
        verbose: 打印进度

    Returns:
        selected_mask: (N_total,) bool array, True=选中
        info: dict 包含统计信息
    """
    N = actions.shape[0]
    selected_mask = np.zeros(N, dtype=bool)

    episode_retention = []
    all_errors = []

    for ep_idx, (start, end) in enumerate(boundaries):
        ep_actions = actions[start:end]
        T = ep_actions.shape[0]  # 此 episode 的帧数

        if T < 10:
            # 太短的 episode，全部保留
            selected_mask[start:end] = True
            episode_retention.append(1.0)
            continue

        # 对 episode 中的每一帧，构建独立的线性模型（隔离目标帧）
        frame_errors = np.full(T, np.inf)
        for t in range(3, T):
            W, intercept = _build_linear_model(ep_actions, t)
            if np.allclose(W, 0) and np.allclose(intercept, 0):
                continue
            pred = ep_actions[t-3:t].flatten() @ W + intercept
            frame_errors[t] = np.linalg.norm(ep_actions[t] - pred)

        # 过滤前 3 帧（无法计算预测误差的帧）：保留
        frame_errors[:3] = np.inf

        all_errors.extend(frame_errors[frame_errors < np.inf].tolist())

        # 自适应阈值
        valid_errors = frame_errors[frame_errors < np.inf]
        if len(valid_errors) == 0:
            selected_mask[start:end] = True
            episode_retention.append(1.0)
            continue

        mu = np.mean(valid_errors)
        sigma = np.std(valid_errors)

        # 二分搜索 k
        k = _bisect_k(valid_errors, mu, sigma, target_ratio, keep_high_error)

        if keep_high_error:
            # v1 (已废弃): 保留高误差帧 — "意外打破预测"
            # ep_mask = frame_errors > (mu + k * sigma)
            ep_mask = frame_errors > (mu + k * sigma)
        else:
            # v2 (当前): 保留低误差帧 — "高效编码原型采样"
            ep_mask = frame_errors < (mu + k * sigma)

        # 前 3 帧无法计算误差，默认保留（避免丢失 episode 开头信息）
        ep_mask[:3] = True

        selected_mask[start:end] = ep_mask
        episode_retention.append(ep_mask.mean())

    retention = selected_mask.mean()
    mode_str = '原型采样(低误差)' if not keep_high_error else '高误差(已废弃)'
    info = {
        'stage': 1,
        'name': f'Predictive Coding ({mode_str})',
        'retention': retention,
        'n_selected': selected_mask.sum(),
        'n_total': N,
        'episode_retention': episode_retention,
        'all_errors': np.array(all_errors),
        'mode': 'prototype' if not keep_high_error else 'surprise',
    }

    if verbose:
        print(f'[Stage 1] 预测编码过滤 ({mode_str}): {info["n_selected"]}/{N} 帧保留 ({retention*100:.1f}%)')

    return selected_mask, info


def _bisect_k(errors, mu, sigma, target_ratio, keep_high_error=False, tol=0.02, max_iter=30):
    """
    二分搜索阈值系数 k，使保留比例接近 target_ratio。

    注意: keep_high_error 改变比较方向:
      - True (v1): 保留 >μ+kσ 的帧, k↑→retention↓
      - False (v2): 保留 <μ+kσ 的帧, k↑→retention↑
    """
    lo, hi = -3.0, 3.0

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if keep_high_error:
            retention = (errors > (mu + mid * sigma)).mean()
        else:
            retention = (errors < (mu + mid * sigma)).mean()

        if abs(retention - target_ratio) < tol:
            return mid
        if retention > target_ratio:
            hi = mid
        else:
            lo = mid

    return (lo + hi) / 2


if __name__ == '__main__':
    # 测试: 用随机数据验证
    np.random.seed(42)
    test_actions = np.cumsum(np.random.randn(1000, 7) * 0.1, axis=0)
    test_boundaries = [(0, 1000)]
    mask, info = predictive_coding_filter(test_actions, test_boundaries, target_ratio=0.375)
    print(f'Retention: {info["retention"]:.3f}')
    print(f'Error stats: mean={info["all_errors"].mean():.4f}, std={info["all_errors"].std():.4f}')
