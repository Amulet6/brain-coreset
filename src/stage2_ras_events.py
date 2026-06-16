"""
Stage 2: RAS 关键事件检测 (Reticular Activating System)

对应脑机制: 网状激活系统 (RAS)
— 根据任务目标过滤背景噪音，聚焦高信息效用瞬间

三种事件类型:
  1. 夹爪突变帧: |gripper_velocity| > 90%分位
  2. 加速度峰值帧: ||a''_t|| > 90%分位
  3. 接触窗口帧: 速度下降率 > 80%分位 AND 夹爪值开始变化

所有阈值基于全局统计分位数。

用法:
  from src.stage2_ras_events import ras_event_detection
  selected_mask, info = ras_event_detection(actions, boundaries)
"""

import numpy as np

# 右臂 7-DoF 中夹爪的索引 (最后一个)
GRIPPER_IDX = 6  # 右臂 gripper 在 7维动作中的位置
GRIPPER_90_PERCENTILE = 90
ACCEL_90_PERCENTILE = 90
CONTACT_80_PERCENTILE = 80


def _compute_velocities(actions):
    """一阶差分: 速度"""
    vel = np.zeros_like(actions)
    vel[1:] = actions[1:] - actions[:-1]
    return vel


def _compute_accelerations(actions):
    """二阶差分: 加速度"""
    acc = np.zeros_like(actions)
    acc[2:] = actions[2:] - 2 * actions[1:-1] + actions[:-2]
    return acc


def ras_event_detection(actions, boundaries, verbose=True):
    """
    阶段2: RAS 关键事件检测

    Args:
        actions: (N_total, 7) 全部训练帧的动作
        boundaries: list of (start, end)
        verbose: 打印进度

    Returns:
        selected_mask: (N_total,) bool array
        info: dict
    """
    N = actions.shape[0]
    selected_mask = np.zeros(N, dtype=bool)

    # 全局统计量
    velocities = _compute_velocities(actions)
    accelerations = _compute_accelerations(actions)

    gripper_vel = np.abs(velocities[:, GRIPPER_IDX])
    acc_magnitude = np.linalg.norm(accelerations, axis=1)
    speed = np.linalg.norm(velocities, axis=1)
    speed_decline = np.zeros(N)
    speed_decline[1:] = speed[:-1] - speed[1:]  # 正值=减速

    # 全局阈值
    gripper_thresh = np.percentile(gripper_vel[gripper_vel > 1e-8], GRIPPER_90_PERCENTILE) if (gripper_vel > 1e-8).any() else 1e-8
    accel_thresh = np.percentile(acc_magnitude[acc_magnitude > 1e-8], ACCEL_90_PERCENTILE) if (acc_magnitude > 1e-8).any() else 1e-8
    decline_thresh = np.percentile(speed_decline[speed_decline > 1e-8], CONTACT_80_PERCENTILE) if (speed_decline > 1e-8).any() else 1e-8

    event_counts = {'gripper': 0, 'accel': 0, 'contact': 0}

    for start, end in boundaries:
        ep_mask = np.zeros(end - start, dtype=bool)

        # 事件1: 夹爪突变
        ep_gripper = gripper_vel[start:end]
        gripper_events = ep_gripper > gripper_thresh
        ep_mask |= gripper_events
        event_counts['gripper'] += gripper_events.sum()

        # 事件2: 加速度峰值
        ep_accel = acc_magnitude[start:end]
        accel_events = ep_accel > accel_thresh
        # 去重: 连续帧只保留第一个
        accel_events = _deduplicate_consecutive(accel_events, window=2)
        ep_mask |= accel_events
        event_counts['accel'] += accel_events.sum()

        # 事件3: 接触窗口 (减速 + 夹爪变化)
        ep_decline = speed_decline[start:end]
        ep_gripper_vel = gripper_vel[start:end]
        contact_events = (ep_decline > decline_thresh) & (ep_gripper_vel > np.percentile(ep_gripper_vel[ep_gripper_vel > 1e-8], 70) if (ep_gripper_vel > 1e-8).any() else 0)
        contact_events = _deduplicate_consecutive(contact_events, window=3)
        ep_mask |= contact_events
        event_counts['contact'] += contact_events.sum()

        selected_mask[start:end] = ep_mask

    retention = selected_mask.mean()
    info = {
        'stage': 2,
        'name': 'RAS Events',
        'retention': retention,
        'n_selected': selected_mask.sum(),
        'n_total': N,
        'event_counts': event_counts,
        'thresholds': {
            'gripper': float(gripper_thresh),
            'accel': float(accel_thresh),
            'contact': float(decline_thresh),
        },
    }

    if verbose:
        print(f'[Stage 2] RAS 事件检测: {info["n_selected"]}/{N} 帧保留 ({retention*100:.1f}%)')
        print(f'  事件: 夹爪突变={event_counts["gripper"]}, 加速度峰值={event_counts["accel"]}, 接触窗口={event_counts["contact"]}')
        print(f'  全局阈值: gripper={gripper_thresh:.4f}, accel={accel_thresh:.4f}, decline={decline_thresh:.4f}')

    return selected_mask, info


def _deduplicate_consecutive(mask, window=2):
    """
    去重连续 True 值，在 window 范围内只保留第一个。

    这模拟了 RAS 的"不应期"——同一事件不会在极短时间内被重复标记。
    """
    indices = np.where(mask)[0]
    if len(indices) <= 1:
        return mask

    result = mask.copy()
    last_kept = indices[0] - window - 1
    for idx in indices:
        if idx - last_kept > window:
            last_kept = idx
        else:
            result[idx] = False

    return result


if __name__ == '__main__':
    np.random.seed(42)
    # 模拟机械臂动作: 大部分时间匀速，偶尔突变
    T = 1000
    actions = np.zeros((T, 7))
    actions[:, :6] = np.cumsum(np.random.randn(T, 6) * 0.05, axis=0)
    actions[:, 6] = np.sin(np.linspace(0, 4*np.pi, T)) * 0.5 + 0.5  # 夹爪周期性开合

    # 插入几个突变点
    actions[200, 0] += 2.0
    actions[500, 6] += 0.8

    boundaries = [(0, T)]
    mask, info = ras_event_detection(actions, boundaries, verbose=True)
