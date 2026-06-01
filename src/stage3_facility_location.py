"""
Stage 3: Facility Location 子模优化多样性采样

对应脑机制: 海马体模式分离 (Pattern Separation)
— 避免记忆高度相似的重复经验，确保记忆库覆盖全面

算法:
  1. 视觉特征 PCA 降维 (512d → 32d) → L2 归一化
  2. 动作特征 L2 归一化
  3. 乘积核: S(i,j) = K_v(v_i,v_j) × K_a(a_i,a_j)
  4. 带宽: σ² = median distance 估计
  5. Facility Location 子模函数: f(C) = Σ_i max_{j∈C} S(i,j)
  6. Lazy Greedy 贪心求解 (1-1/e 近似保证)
  7. 严格输出 全量 10% 的核心集

用法:
  from src.stage3_facility_location import facility_location_selection
  coreset_indices, info = facility_location_selection(features, actions, candidates, n_coreset)
"""

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

PCA_DIM = 32
N_SAMPLES_FOR_BANDWIDTH = 5000


def _estimate_bandwidth(features, n_samples=N_SAMPLES_FOR_BANDWIDTH):
    """
    中位数距离带宽估计。

    Args:
        features: (N, d) L2 归一化特征
        n_samples: 随机采样对数

    Returns:
        sigma_sq: 带宽的平方
    """
    N = features.shape[0]
    if N < 2:
        return 1.0

    n_pairs = min(n_samples, N * (N - 1) // 2)

    # 随机采样索引对
    if N <= 5000:
        idx_i = np.random.choice(N, size=n_pairs, replace=True)
        idx_j = np.random.choice(N, size=n_pairs, replace=True)
    else:
        # 大样本: 随机选点
        subset = np.random.choice(N, size=min(2000, N), replace=False)
        idx_i = np.random.choice(subset, size=n_pairs, replace=True)
        idx_j = np.random.choice(subset, size=n_pairs, replace=True)

    # 确保 i ≠ j
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]

    if len(idx_i) == 0:
        return 1.0

    dists_sq = np.sum((features[idx_i] - features[idx_j]) ** 2, axis=1)
    sigma_sq = np.median(dists_sq)
    sigma_sq = max(sigma_sq, 1e-8)

    return sigma_sq


def _build_similarity_kernel(v_features, a_features, sigma_v_sq, sigma_a_sq):
    """
    乘积核: S(i,j) = K_v(i,j) · K_a(i,j)

    不构建完整矩阵（O(N²) 内存），而是提供按需计算函数。
    """
    def compute_kernel_row(i, candidate_indices=None):
        """计算第 i 个样本到所有候选的相似度"""
        if candidate_indices is None:
            targets_v = v_features
            targets_a = a_features
        else:
            targets_v = v_features[candidate_indices]
            targets_a = a_features[candidate_indices]

        v_dists_sq = np.sum((targets_v - v_features[i]) ** 2, axis=1)
        a_dists_sq = np.sum((targets_a - a_features[i]) ** 2, axis=1)

        K_v = np.exp(-v_dists_sq / sigma_v_sq)
        K_a = np.exp(-a_dists_sq / sigma_a_sq)

        return K_v * K_a

    return compute_kernel_row


def _lazy_greedy_facility_location(v_features, a_features, n_select, verbose=True):
    """
    Lazy Greedy 求解 Facility Location 子模最大化 (正确实现).

    关键修复:
    - 预计算完整核矩阵 K (NxN float32, ~64MB)
    - 初始化 upper_bounds[i] = Σ_j K[i,j] (边际增益上界)
    - heapq 维护 max-heap, 每次 O(log N) 弹出最大值
    - 子模性保证: 真实边际 ≤ 历史边界, lazy 正确

    复杂度: O(N²) 内存 + O(k·N) 计算 (k=1600, N~4000, ~2分钟)
    """
    import heapq

    N = v_features.shape[0]
    if n_select >= N:
        return np.arange(N)

    # 带宽估计
    sigma_v_sq = _estimate_bandwidth(v_features)
    sigma_a_sq = _estimate_bandwidth(a_features)
    if verbose:
        print(f'[Stage 3] Bandwidth: sigma_v^2={sigma_v_sq:.4f}, sigma_a^2={sigma_a_sq:.4f}')

    # ====== 1. 预计算完整核矩阵 (分块, float32) ======
    if verbose:
        print(f'[Stage 3] 预计算 {N}x{N} 核矩阵 (float32, ~{(N*N*4/1024/1024):.0f}MB)...')

    # 预计算每行的 norm (只算一次)
    v_norm2 = np.sum(v_features**2, axis=1).astype(np.float32)  # (N,)
    a_norm2 = np.sum(a_features**2, axis=1).astype(np.float32)  # (N,)

    K = np.zeros((N, N), dtype=np.float32)
    chunk = 400  # 更小的 chunk 降低内存

    for i in range(0, N, chunk):
        end_i = min(i + chunk, N)
        vi = v_features[i:end_i]  # (chunk, 32)
        ai = a_features[i:end_i]  # (chunk, 7)

        # 视觉距离平方
        v_dist = v_norm2[i:end_i, None] + v_norm2[None, :] - 2 * (vi @ v_features.T)
        v_dist = np.maximum(v_dist, 0.0)  # 数值稳定性

        # 动作距离平方
        a_dist = a_norm2[i:end_i, None] + a_norm2[None, :] - 2 * (ai @ a_features.T)
        a_dist = np.maximum(a_dist, 0.0)

        # 乘积核
        K[i:end_i] = np.exp(-v_dist / sigma_v_sq) * np.exp(-a_dist / sigma_a_sq)

    # 清零对角线
    np.fill_diagonal(K, 0.0)

    if verbose:
        print(f'[Stage 3] 核矩阵构建完成, 范围=[{K.min():.4f}, {K.max():.4f}]')

    # ====== 2. 初始化 Lazy Greedy ======
    selected = []
    remaining = set(range(N))
    current_coverage = np.zeros(N, dtype=np.float64)

    # upper_bound[i] = 如果选中 i，它能贡献的最大边际增益 = Σ_j K[i,j]
    upper_bounds = K.sum(axis=1)  # (N,)

    # max-heap: (-bound, i)
    heap = [(-upper_bounds[i], i) for i in range(N)]
    heapq.heapify(heap)

    iterator = range(n_select)
    if verbose:
        iterator = tqdm(iterator, desc='Lazy Greedy 核心集选择')

    for _ in iterator:
        best_idx = None
        best_marginal = -1.0

        while heap:
            neg_bound, i = heapq.heappop(heap)
            bound = -neg_bound

            # 如果 bound 小于当前找到的最佳 marginal, 这个和后面的都不会更好了
            if bound < best_marginal:
                # 把这个放回去, 用 best_idx
                heapq.heappush(heap, (neg_bound, i))
                break

            # 计算真实边际增益
            improvement = np.maximum(0.0, K[i] - current_coverage)
            marginal = improvement.sum()

            # 更新上界
            upper_bounds[i] = marginal

            if marginal > best_marginal:
                # 如果之前的 best_idx 还在, 推回堆
                if best_idx is not None:
                    heapq.heappush(heap, (-best_marginal, best_idx))
                best_marginal = marginal
                best_idx = i
                # 注意: best_idx 不要推回堆, 它将在外层被选中
            else:
                # 重算后不是最佳, 以更新后的 bound 推回堆
                heapq.heappush(heap, (-marginal, i))

                # 如果当前 bound 就是真实值 (且是最大), 可以直接选中
                if bound <= marginal:  # bound 是上界, 不会小于真值
                    break

        if best_idx is None:
            # fallback: 如果堆为空 (极罕见), 全量扫描
            if verbose:
                print('[Stage 3] Warning: heap exhausted, fallback to full scan')
            marginals = np.zeros(N)
            for i in remaining:
                improvement = np.maximum(0.0, K[i] - current_coverage)
                marginals[i] = improvement.sum()
            best_idx = np.argmax(marginals)

        # 选中
        selected.append(best_idx)
        remaining.discard(best_idx)

        # 更新 coverage: coverage[j] = max(coverage[j], K[best_idx, j])
        np.maximum(current_coverage, K[best_idx], out=current_coverage)

    return np.array(selected)


def facility_location_selection(features, actions, candidate_mask, n_coreset, verbose=True):
    """
    阶段3: Facility Location 子模优化

    Args:
        features: (N, 512) 视觉特征 (L2 normalized)
        actions: (N, 7) 动作 (raw, 会在内部 L2 normalize)
        candidate_mask: (N,) bool, 候选池（阶段1+2并集）
        n_coreset: 最终核心集大小 (= 训练总帧数 × 10%)
        verbose: 打印进度

    Returns:
        coreset_indices: (n_coreset,) 选中的帧索引（在全局数组中的位置）
        info: dict
    """
    N = features.shape[0]
    candidate_indices = np.where(candidate_mask)[0]

    if verbose:
        print(f'[Stage 3] 候选池: {len(candidate_indices)}/{N} 帧 ({len(candidate_indices)/N*100:.1f}%)')
        print(f'[Stage 3] 目标核心集: {n_coreset} 帧 ({n_coreset/N*100:.1f}%)')

    # 内存保护: 候选池 > 5000 时随机下采样
    MAX_CAND = 5000
    if len(candidate_indices) > MAX_CAND:
        if verbose:
            print(f'[Stage 3] 候选池 {len(candidate_indices)} > {MAX_CAND}, 随机下采样...')
        np.random.seed(42)
        candidate_indices = np.random.choice(candidate_indices, size=MAX_CAND, replace=False)
        candidate_indices = np.sort(candidate_indices)

    if len(candidate_indices) <= n_coreset:
        if verbose:
            print('[Stage 3] 候选池 ≤ 目标核心集，直接返回全部候选')
        pad = n_coreset - len(candidate_indices)
        if pad > 0:
            # 需要从剩余帧中补充
            remaining = np.setdiff1d(np.arange(N), candidate_indices)
            extra = np.random.choice(remaining, size=pad, replace=False)
            coreset = np.concatenate([candidate_indices, extra])
        else:
            coreset = candidate_indices.copy()
        return coreset, {'stage': 3, 'name': 'Facility Location', 'n_candidate': len(candidate_indices)}

    # PCA 降维 + L2 normalize
    if verbose:
        print('[Stage 3] PCA: 512d → 32d (在候选池上拟合)...')

    candidate_features = features[candidate_indices]
    pca = PCA(n_components=PCA_DIM)
    features_pca = pca.fit_transform(candidate_features)

    # 计算保留方差
    explained_var = pca.explained_variance_ratio_.sum()

    # L2 normalize
    norms = np.linalg.norm(features_pca, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    v_features = features_pca / norms

    # 动作 L2 normalize
    candidate_actions = actions[candidate_indices]
    a_norms = np.linalg.norm(candidate_actions, axis=1, keepdims=True)
    a_norms = np.where(a_norms < 1e-8, 1.0, a_norms)
    a_features = candidate_actions / a_norms

    if verbose:
        print(f'[Stage 3] PCA 保留方差: {explained_var:.1%}')
        print(f'[Stage 3] Lazy Greedy 开始, 目标: {n_coreset} 帧...')

    # Lazy Greedy 选择
    local_indices = _lazy_greedy_facility_location(v_features, a_features, n_coreset, verbose=verbose)

    # 映射回全局索引
    coreset_indices = candidate_indices[local_indices]

    info = {
        'stage': 3,
        'name': 'Facility Location',
        'n_candidate': len(candidate_indices),
        'n_coreset': len(coreset_indices),
        'pca_explained_variance': float(explained_var),
        'sigma_v_sq': float(_estimate_bandwidth(v_features)),
        'sigma_a_sq': float(_estimate_bandwidth(a_features)),
    }

    if verbose:
        print(f'[Stage 3] 核心集: {len(coreset_indices)} 帧 ({len(coreset_indices)/N*100:.1f}%)')

    return coreset_indices, info


if __name__ == '__main__':
    # 测试
    np.random.seed(42)
    N = 1000
    # 模拟特征: 5个聚类
    centers = np.random.randn(5, 512) * 2
    features = np.zeros((N, 512))
    for i in range(N):
        c = i % 5
        features[i] = centers[c] + np.random.randn(512) * 0.3
    # L2 normalize
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    actions = np.random.randn(N, 7)

    # 候选池: 所有帧
    candidate_mask = np.ones(N, dtype=bool)
    n_coreset = 100  # 10%

    coreset, info = facility_location_selection(features, actions, candidate_mask, n_coreset, verbose=True)
    print(f'核心集: {len(coreset)} 帧')
    print(f'PCA 保留方差: {info["pca_explained_variance"]:.1%}')
