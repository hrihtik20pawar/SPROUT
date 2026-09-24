"""Count parameters for all SPROUT model components"""
import sys
sys.path.insert(0, '.')

import torch

# Inline model definitions to avoid import issues
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, backbone='efficientnet_v2_s', pretrained=True):
        super().__init__()
        if backbone == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet50(weights=weights)
            self.feature_dim = base.fc.in_features
            self.backbone = nn.Sequential(*list(base.children())[:-1])
        elif backbone == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            self.feature_dim = base.classifier[1].in_features
            self.backbone = nn.Sequential(*list(base.children())[:-1])
        elif backbone == 'efficientnet_v2_s':
            weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.efficientnet_v2_s(weights=weights)
            self.feature_dim = base.classifier[1].in_features
            self.backbone = nn.Sequential(*list(base.children())[:-1])
        else:
            raise ValueError(f"Unsupported: {backbone}")

    def forward(self, x):
        features = self.backbone(x)
        return features.view(features.size(0), -1)

    def get_feature_dim(self):
        return self.feature_dim


class EmbeddingNetwork(nn.Module):
    def __init__(self, input_dim, embed_dim=128, hidden_dims=(512, 256), dropout_rate=0.3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(p=dropout_rate))
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.Linear(hidden_dims[-1], embed_dim))
        self.embedding_layers = nn.Sequential(*layers)

    def forward(self, x):
        embeddings = self.embedding_layers(x)
        return F.normalize(embeddings, p=2, dim=1)


class PrototypeModule(nn.Module):
    def __init__(self, embed_dim=128, num_refinement_steps=3, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_refinement_steps = num_refinement_steps
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        self.attention = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1)
        )
        self.refinement_network = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        self.attn_temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x


def count_params(name, module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in module.buffers())
    print(f"  {name:30s} | Total: {total:>12,} | Trainable: {trainable:>12,} | Buffers: {buffers:>10,}")
    return total, trainable


print("=" * 80)
print("SPROUT MODEL PARAMETER COUNT")
print("=" * 80)

for backbone in ['efficientnet_v2_s', 'resnet50', 'efficientnet_b0']:
    print(f"\n--- Backbone: {backbone} ---")

    fe = FeatureExtractor(backbone=backbone)
    feature_dim = fe.get_feature_dim()
    print(f"  Feature dim: {feature_dim}")

    en = EmbeddingNetwork(input_dim=feature_dim, embed_dim=128, hidden_dims=(512, 256), dropout_rate=0.3)
    pm = PrototypeModule(embed_dim=128, num_refinement_steps=3, num_heads=4)

    total_fe, _ = count_params("FeatureExtractor", fe)
    total_en, _ = count_params("EmbeddingNetwork", en)
    total_pm, _ = count_params("PrototypeModule", pm)
    temp_params = 1
    print(f"  {'Temperature (scalar)':30s} | Total: {temp_params:>12,} | Trainable: {temp_params:>12,}")

    total = total_fe + total_en + total_pm + temp_params
    print(f"  {'_' * 78}")
    print(f"  {'TOTAL':30s} | Total: {total:>12,}")

    mem_mb = total * 4 / 1024 / 1024
    print(f"  Memory (float32): {mem_mb:.2f} MB | Memory (float16): {mem_mb/2:.2f} MB")
    print()

    # Optimizer memory (Adam: 2 momentum buffers per param)
    optimizer_mem = total * 4 * 3 / 1024 / 1024  # params + m + v
    print(f"  Optimizer (Adam) memory: {optimizer_mem:.2f} MB")
    print(f"  Full checkpoint (model+optim): {optimizer_mem + mem_mb:.2f} MB")
    break
