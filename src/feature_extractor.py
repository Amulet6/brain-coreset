"""
特征提取模块：支持 Frozen ResNet-18 和 CLIP ViT-B-32

功能：
  1. ResNet18Encoder: 冻结 ResNet-18 (ImageNet 预训练), 512d
  2. CLIPEncoder: 冻结 CLIP ViT-B-32 (LAION-2B 预训练), 512d
  3. Multi-Crop TTA: 原图 + 水平翻转 + 中心裁剪 → 平均特征
  4. 批量处理 + GPU 加速
  5. 缓存提取的特征到磁盘

用法：
  from src.feature_extractor import FeatureExtractor
  extractor = FeatureExtractor(device='cuda', encoder_type='clip')  # 或 'resnet'
  features = extractor.extract(all_frames)  # (N, 512)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

# 全局配置
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cache')
FEATURE_DIM = 512
PCA_DIM = 32
BATCH_SIZE = 64  # CLIP ViT 可以适当调小
CLIP_BATCH_SIZE = 32


class ResNet18Encoder(nn.Module):
    """冻结的 ResNet-18，输出 512-d 特征向量"""

    def __init__(self):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
        )
        self.avgpool = model.avgpool
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            feat = self.backbone(x)
            feat = self.avgpool(feat)
            feat = feat.view(feat.size(0), -1)
            feat = nn.functional.normalize(feat, p=2, dim=1)
        return feat


class CLIPEncoder(nn.Module):
    """冻结的 CLIP ViT-B-32，输出 512-d 视觉特征向量"""

    def __init__(self):
        super().__init__()
        import open_clip
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

    def forward(self, x):
        """x: (B, 3, H, W) — 已经过 CLIP 预处理的图像"""
        with torch.no_grad():
            feat = self.clip_model.encode_image(x, normalize=True)
        return feat


def get_language_embedding(text="transfer the red cube from one arm to the other", device='cuda'):
    """
    用 CLIP Text Encoder 编码语言指令, 返回 512d 向量.

    ALOHA 数据集中所有 Episode 为同一任务, 语言嵌入为常量.
    """
    import open_clip
    device = device if torch.cuda.is_available() else 'cpu'
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        text_tokens = tokenizer([text]).to(device)
        text_feat = model.encode_text(text_tokens, normalize=True)
    return text_feat.cpu().numpy().flatten().astype(np.float32)


class FeatureExtractor:
    """
    视觉特征提取器

    encoder_type: 'resnet' (默认) 或 'clip'
    - Multi-Crop TTA: 对每帧做 3 种变换取平均
    - 批量 GPU 处理
    - 自动缓存
    """

    def __init__(self, device='cuda', encoder_type='resnet'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.encoder_type = encoder_type

        if encoder_type == 'clip':
            self.encoder = CLIPEncoder().to(self.device)
            self.clip_pp = self.encoder.clip_preprocess  # CLIP 内置预处理

            # CLIP TTA: 3 路 (原图 / 翻转 / 中心裁剪)
            self.tta_transforms = [
                lambda f: self.clip_pp(T.ToPILImage()(f.transpose(1, 2, 0))),
                lambda f: self.clip_pp(T.RandomHorizontalFlip(p=1.0)(T.ToPILImage()(f.transpose(1, 2, 0)))),
                lambda f: self.clip_pp(T.CenterCrop(400)(T.ToPILImage()(f.transpose(1, 2, 0)))),
            ]
        else:
            self.encoder = ResNet18Encoder().to(self.device)
            self.normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            self.tta_transforms = [
                T.Compose([self._chw2hwc, T.ToPILImage(), T.Resize(256), T.CenterCrop(224), T.ToTensor(), self.normalize]),
                T.Compose([self._chw2hwc, T.ToPILImage(), T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), self.normalize]),
                T.Compose([self._chw2hwc, T.ToPILImage(), T.Resize(256), T.Resize(224), T.ToTensor(), self.normalize]),
            ]

        label = f'[FeatureExtractor] {encoder_type.upper()}'
        if self.device == 'cuda':
            label += f' - GPU: {torch.cuda.get_device_name(0)}'
        else:
            label += ' - CPU'
        print(label)

    @staticmethod
    def _chw2hwc(frame):
        """numpy (3,H,W) → (H,W,3)"""
        return frame.transpose(1, 2, 0)

    def extract(self, frames, use_cache=True, cache_name='train_features', verbose=True):
        """
        从原始帧提取特征。

        Args:
            frames: (N, 3, H, W) uint8 numpy array (CHW 格式)
            use_cache: 是否使用缓存
            cache_name: 缓存文件名前缀
            verbose: 是否打印进度

        Returns:
            features: (N, 512) float32 numpy array
        """
        cache_path = os.path.join(CACHE_DIR, f'{cache_name}.npy')

        if use_cache and os.path.exists(cache_path):
            if verbose:
                print(f'[FeatureExtractor] 从缓存加载特征: {cache_path}')
            return np.load(cache_path)

        batch_sz = CLIP_BATCH_SIZE if self.encoder_type == 'clip' else BATCH_SIZE

        if verbose:
            print(f'[FeatureExtractor] 提取特征 ({self.encoder_type.upper()}), 共 {len(frames)} 帧...')

        n_frames = len(frames)
        features = np.zeros((n_frames, FEATURE_DIM), dtype=np.float32)

        for start in tqdm(range(0, n_frames, batch_sz),
                          desc=f'提取{self.encoder_type.upper()}特征', disable=not verbose):
            end = min(start + batch_sz, n_frames)
            batch_frames = frames[start:end]

            # Multi-Crop TTA: 3 路取平均
            batch_feat = np.zeros((end - start, FEATURE_DIM), dtype=np.float32)
            for tf in self.tta_transforms:
                tensors = [tf(frame) for frame in batch_frames]
                batch = torch.stack(tensors).to(self.device)
                feat = self.encoder(batch).cpu().numpy()
                batch_feat += feat

            batch_feat /= len(self.tta_transforms)
            # 再次 L2 normalize (TTA 平均后)
            norm = np.linalg.norm(batch_feat, axis=1, keepdims=True)
            norm = np.where(norm < 1e-8, 1.0, norm)
            batch_feat /= norm

            features[start:end] = batch_feat

        np.save(cache_path, features)
        if verbose:
            print(f'[FeatureExtractor] 特征已缓存到: {cache_path}')

        return features

    def extract_single_frame(self, frame):
        """提取单帧特征"""
        self.encoder.eval()
        tensors = [tf(frame) for tf in self.tta_transforms]
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            feats = self.encoder(batch).cpu().numpy()
        feat = feats.mean(axis=0)
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        return feat


# ---- 保留兼容旧接口 ----
def batch_extract_features(frames, encoder, device='cuda', batch_size=64, verbose=True):
    """简化版批量特征提取（不使用 TTA）"""
    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    n_frames = len(frames)
    features = np.zeros((n_frames, FEATURE_DIM), dtype=np.float32)

    iterator = tqdm(range(0, n_frames, batch_size), desc='快速特征提取') if verbose else range(0, n_frames, batch_size)
    for start in iterator:
        end = min(start + batch_size, n_frames)
        tensors = [preprocess(frames[i]) for i in range(start, end)]
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            feat = encoder(batch).cpu().numpy()
        norm = np.linalg.norm(feat, axis=1, keepdims=True)
        norm = np.where(norm < 1e-8, 1.0, norm)
        feat /= norm
        features[start:end] = feat
    return features


if __name__ == '__main__':
    print('[Test] ResNet-18:')
    rn = ResNet18Encoder()
    print(f'  输出: {rn(torch.randn(4,3,224,224)).shape}')

    print('[Test] CLIP:')
    clip = CLIPEncoder()
    print(f'  输出: {clip(torch.randn(4,3,224,224)).shape}')
    print('通过')
