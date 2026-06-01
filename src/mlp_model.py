"""
MLP 动作预测模型：轻量级多层感知机

架构: 512 → 256 → 128 → 7 (ReLU)
用法:
  from src.mlp_model import MLP, train_mlp, evaluate_mlp
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# 全局配置
INPUT_DIM = 512
HIDDEN_DIMS = [256, 128]
OUTPUT_DIM = 7
LR = 1e-3
EPOCHS = 100
BATCH_SIZE = 64
SEED = 42


def set_seed(seed=SEED):
    """固定随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    """
    三层 MLP: 512 (ResNet-18 特征) → 256 → 128 → 7 (动作)

    对应题目要求："轻量级的多层感知机（MLP）"
    """

    def __init__(self, input_dim=INPUT_DIM, hidden_dims=None, output_dim=OUTPUT_DIM, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, 512) 视觉特征
        Returns:
            (B, 7) 预测动作
        """
        return self.net(x)

    def count_parameters(self):
        """统计参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DualHeadMLP(nn.Module):
    """
    双头 MLP: 前6维回归(MSE) + 夹爪分类(BCE)

    对应问题2修复: 夹爪是二值离散变量, MSE 不合适.
    架构: input → shared hidden layers → split
          ├─ reg_head: 6-dim (joint positions)
          └─ cls_head: 1-dim + Sigmoid (gripper open/close)
    """

    def __init__(self, input_dim=INPUT_DIM, hidden_dims=None, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS

        # 共享层
        shared = []
        prev = input_dim
        for h in hidden_dims:
            shared.append(nn.Linear(prev, h))
            shared.append(nn.ReLU())
            shared.append(nn.Dropout(dropout))
            prev = h
        self.shared = nn.Sequential(*shared)

        # 回归头 (前6维)
        self.reg_head = nn.Linear(hidden_dims[-1], 6)
        # 分类头 (夹爪)
        self.cls_head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        feat = self.shared(x)
        reg = self.reg_head(feat)         # (B, 6)
        cls_logit = self.cls_head(feat)   # (B, 1)
        return reg, cls_logit

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_mlp(model, train_features, train_actions, val_features=None, val_actions=None,
              epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, device='cuda', verbose=True):
    """
    训练 MLP。

    Args:
        model: MLP 实例
        train_features: (N_train, 512) numpy array
        train_actions: (N_train, 7) numpy array
        val_features: (N_val, 512) 可选验证集
        val_actions: (N_val, 7) 可选验证集
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        device: 'cuda' or 'cpu'
        verbose: 是否打印进度

    Returns:
        history: dict with 'train_loss', 'val_loss' lists
    """
    set_seed(SEED)

    device = device if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # 构建 DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(train_features),
        torch.FloatTensor(train_actions)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if val_features is not None:
        val_dataset = TensorDataset(
            torch.FloatTensor(val_features),
            torch.FloatTensor(val_actions)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )

    history = {'train_loss': [], 'val_loss': []}

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc='MLP 训练')

    for epoch in iterator:
        # 训练
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # 验证
        if val_features is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    pred = model(batch_x)
                    loss = criterion(pred, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
            val_loss /= len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            scheduler.step(val_loss)
        else:
            scheduler.step(train_loss)

        if verbose and (epoch + 1) % 20 == 0:
            v_str = f', val_loss={val_loss:.6f}' if val_features is not None else ''
            if isinstance(iterator, tqdm):
                iterator.set_postfix({'train': f'{train_loss:.6f}', **({'val': f'{val_loss:.6f}'} if val_features is not None else {})})

    return history


def evaluate_mlp(model, test_features, test_actions, device='cuda', return_per_dim=False):
    """
    评估 MLP 在测试集上的 MSE。

    Args:
        model: 训练好的 MLP
        test_features: (N_test, 512)
        test_actions: (N_test, 7)
        device: 'cuda' or 'cpu'
        return_per_dim: 是否返回每个维度的 MSE

    Returns:
        mse: float (整体 MSE)
        per_dim_mse: (7,) numpy array (if return_per_dim=True)
        predictions: (N_test, 7) 预测值
    """
    device = device if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    test_dataset = TensorDataset(
        torch.FloatTensor(test_features),
        torch.FloatTensor(test_actions)
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    criterion = nn.MSELoss(reduction='sum')
    total_loss = 0.0
    total_samples = 0
    all_preds = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            total_loss += loss.item()
            total_samples += batch_x.size(0)
            all_preds.append(pred.cpu().numpy())

    mse = total_loss / total_samples
    predictions = np.concatenate(all_preds, axis=0)

    if return_per_dim:
        per_dim_mse = np.mean((predictions - test_actions) ** 2, axis=0)
        return mse, per_dim_mse, predictions

    return mse


# ============================================================
# Dual-Head MLP: 前6维回归 + 夹爪分类
# ============================================================
GRIPPER_IDX = 6  # 夹爪在7维动作中的索引

def train_dual_mlp(model, train_features, train_actions,
                   val_features=None, val_actions=None,
                   epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                   reg_weight=0.86, device='cuda', verbose=True):
    """
    训练双头 MLP (6-d 回归 + 夹爪分类).

    Args:
        reg_weight: 回归损失权重 (0.86 ≈ 6/7)
        train_actions: (N, 7) — 第7维是夹爪, 二值化后用于 BCE
    """
    set_seed(SEED)
    device = device if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # 拆分: 前6维回归, 第7维分类
    reg_y = torch.FloatTensor(train_actions[:, :GRIPPER_IDX])
    # 夹爪: 归一化后正值→1(闭合), 负值→0(张开)
    gripper_y = torch.FloatTensor((train_actions[:, GRIPPER_IDX] > 0).astype(np.float32)).unsqueeze(1)
    train_dataset = TensorDataset(torch.FloatTensor(train_features), reg_y, gripper_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_features is not None:
        val_reg = torch.FloatTensor(val_actions[:, :GRIPPER_IDX])
        val_grip = torch.FloatTensor((val_actions[:, GRIPPER_IDX] > 0).astype(np.float32)).unsqueeze(1)
        val_dataset = TensorDataset(torch.FloatTensor(val_features), val_reg, val_grip)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    history = {'train_loss': [], 'val_loss': []}
    cls_weight = 1.0 - reg_weight

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc='Dual-MLP 训练')

    for epoch in iterator:
        model.train()
        train_loss = 0.0
        for batch_x, batch_reg, batch_cls in train_loader:
            batch_x, batch_reg, batch_cls = batch_x.to(device), batch_reg.to(device), batch_cls.to(device)
            optimizer.zero_grad()
            reg_pred, cls_logit = model(batch_x)
            loss_reg = mse_criterion(reg_pred, batch_reg)
            loss_cls = bce_criterion(cls_logit, batch_cls)
            loss = reg_weight * loss_reg + cls_weight * loss_cls
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_reg, batch_cls in val_loader:
                    batch_x, batch_reg, batch_cls = batch_x.to(device), batch_reg.to(device), batch_cls.to(device)
                    reg_pred, cls_logit = model(batch_x)
                    loss = reg_weight * mse_criterion(reg_pred, batch_reg) + cls_weight * bce_criterion(cls_logit, batch_cls)
                    val_loss += loss.item() * batch_x.size(0)
            val_loss /= len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            scheduler.step(val_loss)
        else:
            scheduler.step(train_loss)

    return history


def evaluate_dual_mlp(model, test_features, test_actions, device='cuda'):
    """
    评估双头 MLP.

    Returns:
        mse_6d: 前6维联合 MSE
        gripper_accuracy: 夹爪准确率
        gripper_f1: 夹爪 F1 score
        predictions: (N, 7) [6-d reg, gripper prob]
    """
    from sklearn.metrics import accuracy_score, f1_score
    device = device if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    reg_y = torch.FloatTensor(test_actions[:, :GRIPPER_IDX])
    gripper_y = (test_actions[:, GRIPPER_IDX] > 0).astype(np.int32)
    test_dataset = TensorDataset(torch.FloatTensor(test_features), reg_y, torch.FloatTensor(gripper_y))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    all_reg_preds, all_cls_probs, all_cls_labels = [], [], []

    mse_criterion = nn.MSELoss(reduction='sum')
    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_reg, batch_cls in test_loader:
            batch_x = batch_x.to(device)
            reg_pred, cls_logit = model(batch_x)
            loss = mse_criterion(reg_pred, batch_reg.to(device))
            total_mse += loss.item()
            total_samples += batch_x.size(0)
            all_reg_preds.append(reg_pred.cpu().numpy())
            all_cls_probs.append(torch.sigmoid(cls_logit).cpu().numpy())
            all_cls_labels.append(batch_cls.numpy())

    mse_6d = total_mse / (total_samples * 6)  # 6个回归维度
    reg_preds = np.concatenate(all_reg_preds, axis=0)
    cls_probs = np.concatenate(all_cls_probs, axis=0).flatten()
    cls_labels = np.concatenate(all_cls_labels, axis=0).flatten()
    cls_preds = (cls_probs > 0.5).astype(np.int32)

    acc = accuracy_score(cls_labels, cls_preds)
    f1 = f1_score(cls_labels, cls_preds, zero_division=0)

    # 拼接回7维用于 per-dim MSE
    all_preds = np.hstack([reg_preds, cls_preds.reshape(-1, 1).astype(np.float32)])

    return mse_6d, acc, f1, all_preds


if __name__ == '__main__':
    # 测试
    set_seed(42)
    model = MLP()
    print(f'MLP 参数量: {model.count_parameters():,}')
    dummy_x = torch.randn(16, 512)
    dummy_y = model(dummy_x)
    print(f'输入: {dummy_x.shape} → 输出: {dummy_y.shape}')

    model2 = DualHeadMLP(input_dim=2048)
    print(f'DualHeadMLP 参数量: {model2.count_parameters():,}')
    r, c = model2(torch.randn(4, 2048))
    print(f'DualHead: reg={r.shape}, cls={c.shape}')
    print('MLP 测试通过')
