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


if __name__ == '__main__':
    # 测试
    set_seed(42)
    model = MLP()
    print(f'MLP 参数量: {model.count_parameters():,}')
    dummy_x = torch.randn(16, 512)
    dummy_y = model(dummy_x)
    print(f'输入: {dummy_x.shape} → 输出: {dummy_y.shape}')
    print('MLP 测试通过')
