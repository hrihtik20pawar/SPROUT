"""
SPROUT Enhanced Training - VS Code Version
All checkpoints saved permanently. Enhanced model with multi-head attention, warmup, AMP.

Fixes applied:
- Double forward pass removed (support features computed once)
- Scheduler combination fixed (SequentialLR)
- --gpu flag fixed (negatable)
- Checkpoint size reduced (save best only, not every epoch full state)
- torch.load weights_only=True
- Removed unused num_classes param
- Paddy test set graceful handling
- Per-episode seed reproducibility
- Removed unused imports
- autocast updated to torch.amp
- torch.cuda.empty_cache() called periodically
- Dead code removed
"""
import os
import random
import time
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    DATA_DIR = "./data/cassava"
    OUTPUT_DIR = "./results/cassava_enhanced"
    BACKBONE = 'efficientnet_v2_s'
    EMBED_DIM = 128
    HIDDEN_DIMS = [512, 256]
    NUM_REFINEMENT_STEPS = 3
    TEMPERATURE = 10.0
    DROPOUT_RATE = 0.3
    NUM_HEADS = 4
    N_WAY = 5
    K_SHOT = 5
    N_QUERY = 15
    NUM_EPISODES = 100
    NUM_EPOCHS = 40
    LR = 0.001
    WEIGHT_DECAY = 1e-5
    WARMUP_EPOCHS = 3
    ALPHA = 0.5
    BETA = 0.3
    GAMMA = 0.2
    MARGIN = 1.0
    LABEL_SMOOTHING = 0.1
    NUM_TEST_EPISODES = 200
    TEST_SHOTS = [1, 3, 5, 10]
    USE_AMP = True
    SEED = 42
    SAVE_BEST_ONLY = True  # Save only best checkpoint + final (saves disk)
    MAX_CHECKPOINTS = 3    # Keep top-K checkpoints if SAVE_BEST_ONLY=False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================
class LeafDiseaseDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split_dir = self.root_dir / split

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.classes = sorted([d.name for d in self.split_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []

        for class_name in self.classes:
            class_dir = self.split_dir / class_name
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.bmp', '*.tiff']:
                for img_path in class_dir.glob(f'**/{ext}'):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

        self.labels = torch.tensor(self.labels)
        self.indices_by_class = {}
        for class_idx in range(len(self.classes)):
            self.indices_by_class[class_idx] = torch.where(self.labels == class_idx)[0].tolist()

        print(f"  {split}: {len(self.image_paths)} images across {len(self.classes)} classes")
        for cls_name, count in self.get_class_counts().items():
            print(f"    {cls_name}: {count} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = torch.zeros((3, 224, 224))
        return image, label

    def get_classes(self):
        return self.classes

    def get_class_counts(self):
        return {self.classes[i]: len(self.indices_by_class[i]) for i in range(len(self.classes))}


# ============================================================
# MODEL COMPONENTS
# ============================================================
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
        elif backbone == 'efficientnet_v2_m':
            weights = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.efficientnet_v2_m(weights=weights)
            self.feature_dim = base.classifier[1].in_features
            self.backbone = nn.Sequential(*list(base.children())[:-1])
        elif backbone == 'efficientnet_v2_l':
            weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.efficientnet_v2_l(weights=weights)
            self.feature_dim = base.classifier[1].in_features
            self.backbone = nn.Sequential(*list(base.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

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

    def generate_initial_prototypes(self, support_embeddings, support_labels):
        classes = torch.unique(support_labels)
        prototypes = []
        for c in classes:
            class_mask = (support_labels == c)
            class_embeddings = support_embeddings[class_mask]
            if len(class_embeddings) > 0:
                prototype = torch.mean(class_embeddings, dim=0)
            else:
                prototype = torch.zeros(support_embeddings.size(1), device=support_embeddings.device)
            prototypes.append(prototype)
        return torch.stack(prototypes)

    def compute_attention_weights(self, support_embeddings, prototype, support_labels, class_idx):
        class_mask = (support_labels == class_idx)
        class_embeddings = support_embeddings[class_mask]
        if len(class_embeddings) == 0:
            return None
        prototype_expanded = prototype.unsqueeze(0).expand(class_embeddings.size(0), -1)
        attention_input = torch.cat([class_embeddings, prototype_expanded], dim=1)
        attention_scores = self.attention(attention_input)
        attention_scores = attention_scores / self.attn_temperature
        attention_weights = F.softmax(attention_scores, dim=0)
        return attention_weights, class_embeddings

    def refine_prototype(self, prototype, support_embeddings, support_labels, class_idx):
        attention_result = self.compute_attention_weights(support_embeddings, prototype, support_labels, class_idx)
        if attention_result is None:
            return prototype
        attention_weights, class_embeddings = attention_result
        weighted_avg = torch.sum(attention_weights * class_embeddings, dim=0)
        class_embeddings_seq = class_embeddings.unsqueeze(0)
        prototype_query = prototype.unsqueeze(0).unsqueeze(0)
        attn_output, _ = self.multihead_attn(
            query=prototype_query, key=class_embeddings_seq, value=class_embeddings_seq
        )
        attn_refined = attn_output.squeeze(0).squeeze(0)
        refinement_input = torch.cat([prototype, weighted_avg + attn_refined], dim=0)
        refinement_delta = self.refinement_network(refinement_input.unsqueeze(0)).squeeze(0)
        refined_prototype = prototype + refinement_delta
        return F.normalize(refined_prototype, p=2, dim=0)

    def forward(self, support_embeddings, support_labels):
        if len(support_embeddings) == 0:
            return torch.zeros((0, self.embed_dim), device=support_embeddings.device)
        prototypes = self.generate_initial_prototypes(support_embeddings, support_labels)
        classes = torch.unique(support_labels)
        for _ in range(self.num_refinement_steps):
            refined_prototypes = []
            for i, c in enumerate(classes):
                if i < len(prototypes):
                    refined_prototypes.append(self.refine_prototype(prototypes[i], support_embeddings, support_labels, c))
            prototypes = torch.stack(refined_prototypes) if refined_prototypes else prototypes
        return prototypes


class SPROUT(nn.Module):
    def __init__(self, backbone='efficientnet_v2_s', embed_dim=128,
                 hidden_dims=(512, 256), num_refinement_steps=3, temperature=10.0,
                 dropout_rate=0.3, num_heads=4):
        super().__init__()
        self.feature_extractor = FeatureExtractor(backbone=backbone)
        feature_dim = self.feature_extractor.get_feature_dim()
        self.embedding_network = EmbeddingNetwork(
            input_dim=feature_dim, embed_dim=embed_dim,
            hidden_dims=hidden_dims, dropout_rate=dropout_rate
        )
        self.prototype_module = PrototypeModule(
            embed_dim=embed_dim, num_refinement_steps=num_refinement_steps, num_heads=num_heads
        )
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

    def forward(self, query_images, support_images=None, support_labels=None):
        query_features = self.feature_extractor(query_images)
        query_embeddings = self.embedding_network(query_features)
        if support_images is not None and support_labels is not None:
            support_features = self.feature_extractor(support_images)
            support_embeddings = self.embedding_network(support_features)
            prototypes = self.prototype_module(support_embeddings, support_labels)
            logits = -self.compute_distances(query_embeddings, prototypes)
            return logits, prototypes, support_embeddings
        return query_embeddings

    def compute_distances(self, embeddings, prototypes):
        embeddings_expanded = embeddings.unsqueeze(1)
        prototypes_expanded = prototypes.unsqueeze(0)
        distances = torch.sum((embeddings_expanded - prototypes_expanded) ** 2, dim=2)
        temperature = torch.exp(self.log_temperature)
        return distances / temperature

    def extract_embeddings(self, images):
        self.eval()
        with torch.no_grad():
            features = self.feature_extractor(images)
            embeddings = self.embedding_network(features)
        return embeddings


# ============================================================
# LOSS FUNCTION
# ============================================================
class SPROUTLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2, margin=1.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.margin = margin
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets, prototypes, initial_prototypes, support_embeddings, support_labels):
        proto_loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
        refine_loss = self.alpha * torch.mean(1 - F.cosine_similarity(prototypes, initial_prototypes, dim=1))
        classes = torch.unique(support_labels)
        intra_loss = torch.tensor(0.0, device=logits.device)
        for c in classes:
            mask = (support_labels == c)
            if torch.sum(mask) > 1:
                class_emb = support_embeddings[mask]
                centroid = torch.mean(class_emb, dim=0, keepdim=True)
                intra_loss = intra_loss + torch.var(torch.sum((class_emb - centroid) ** 2, dim=1))
        intra_loss = self.beta * intra_loss / len(classes) if len(classes) > 0 else intra_loss
        inter_loss = torch.tensor(0.0, device=logits.device)
        if prototypes.size(0) > 1:
            distances = torch.cdist(prototypes, prototypes, p=2)
            mask_upper = torch.triu(torch.ones_like(distances), diagonal=1) == 1
            inter_loss = self.gamma * F.relu(self.margin - distances[mask_upper]).mean()
        total_loss = proto_loss + refine_loss + intra_loss + inter_loss
        return total_loss, {
            'proto_loss': proto_loss.item(),
            'refine_loss': refine_loss.item(),
            'intra_loss': intra_loss.item(),
            'inter_loss': inter_loss.item(),
            'total_loss': total_loss.item()
        }


# ============================================================
# EPISODE CREATION (with per-episode seed)
# ============================================================
def create_episode(dataset, n_way, k_shot, n_query, device, episode_seed=None):
    if episode_seed is not None:
        rng_state = random.getstate()
        np_rng_state = np.random.get_state()
        random.seed(episode_seed)
        np.random.seed(episode_seed)

    indices_by_class = dataset.indices_by_class
    available_classes = [c for c in indices_by_class if len(indices_by_class[c]) >= k_shot + n_query]
    if len(available_classes) < n_way:
        available_classes = [c for c in indices_by_class if len(indices_by_class[c]) >= 2]
        n_way = min(n_way, len(available_classes))
    if len(available_classes) < 2:
        if episode_seed is not None:
            random.setstate(rng_state)
            np.random.set_state(np_rng_state)
        raise ValueError(f"Not enough classes with sufficient samples. Have {len(available_classes)}, need at least 2.")
    selected_classes = random.sample(available_classes, n_way)
    support_images, support_labels = [], []
    query_images, query_labels = [], []
    for new_label, class_idx in enumerate(selected_classes):
        indices = indices_by_class[class_idx]
        shuffled = indices.copy()
        random.shuffle(shuffled)
        for idx in shuffled[:k_shot]:
            img, _ = dataset[idx]
            support_images.append(img)
            support_labels.append(new_label)
        for idx in shuffled[k_shot:k_shot + n_query]:
            img, _ = dataset[idx]
            query_images.append(img)
            query_labels.append(new_label)

    if episode_seed is not None:
        random.setstate(rng_state)
        np.random.set_state(np_rng_state)

    support_images = torch.stack(support_images).to(device)
    support_labels = torch.tensor(support_labels).to(device)
    query_images = torch.stack(query_images).to(device)
    query_labels = torch.tensor(query_labels).to(device)
    return support_images, support_labels, query_images, query_labels


# ============================================================
# TRAINING (fixed: no double forward pass)
# ============================================================
def train_one_epoch(model, criterion, optimizer, scaler, dataset, config, device, epoch):
    model.train()
    episode_accuracies, episode_losses = [], []
    pbar = tqdm(range(config.NUM_EPISODES), desc="  Episodes", leave=False)
    for ep in pbar:
        optimizer.zero_grad()
        # FIX: Use per-episode seed for reproducibility
        episode_seed = config.SEED * 10000 + epoch * 100 + ep
        support_images, support_labels, query_images, query_labels = create_episode(
            dataset, config.N_WAY, config.K_SHOT, config.N_QUERY, device, episode_seed=episode_seed
        )
        use_amp = config.USE_AMP and torch.cuda.is_available()
        with autocast('cuda' if use_amp else 'cpu', enabled=use_amp):
            # FIX: Single forward pass - model now returns support_embeddings
            logits, prototypes, support_embeddings = model(query_images, support_images, support_labels)
            # Compute initial prototypes from already-computed support_embeddings (no extra forward pass)
            initial_prototypes = model.prototype_module.generate_initial_prototypes(
                support_embeddings.detach(), support_labels
            )
            loss, loss_components = criterion(
                logits, query_labels, prototypes, initial_prototypes,
                support_embeddings.detach(), support_labels
            )
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        _, predicted = torch.max(logits.data, 1)
        accuracy = (predicted == query_labels).float().mean().item()
        episode_accuracies.append(accuracy)
        episode_losses.append(loss_components['total_loss'])
        pbar.set_postfix(acc=f"{accuracy:.3f}", loss=f"{loss_components['total_loss']:.4f}")
    return np.mean(episode_accuracies), np.mean(episode_losses)


# ============================================================
# EVALUATION
# ============================================================
def evaluate(model, test_dataset, config, device, shot=None):
    model.eval()
    k_shot = shot if shot else config.K_SHOT
    if len(test_dataset) == 0:
        print(f"  WARNING: Test dataset is empty, skipping {k_shot}-shot evaluation")
        return 0.0, 0.0, [], []
    all_preds, all_labels, episode_accs = [], [], []
    for _ in tqdm(range(config.NUM_TEST_EPISODES), desc=f"  Evaluating ({k_shot}-shot)", leave=False):
        support_images, support_labels, query_images, query_labels = create_episode(
            test_dataset, config.N_WAY, k_shot, config.N_QUERY, device
        )
        with torch.no_grad():
            logits, _, _ = model(query_images, support_images, support_labels)
        _, predicted = torch.max(logits.data, 1)
        episode_accs.append((predicted == query_labels).float().mean().item())
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(query_labels.cpu().numpy())
    return np.mean(episode_accs), np.std(episode_accs), all_preds, all_labels


# ============================================================
# CHECKPOINT MANAGEMENT (disk-space efficient)
# ============================================================
def save_checkpoint(model, optimizer, epoch, accuracy, loss, output_dir, is_best=False):
    if is_best:
        torch.save(model.state_dict(), os.path.join(output_dir, 'sprout_best.pth'))
    # Always save latest for resume capability (lightweight - no optimizer state)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'loss': loss,
    }, os.path.join(output_dir, 'sprout_latest.pth'))


def cleanup_old_checkpoints(output_dir, keep_latest=True, max_keep=3):
    checkpoints = sorted(Path(output_dir).glob('checkpoint_epoch_*.pth'))
    if len(checkpoints) > max_keep:
        for ckpt in checkpoints[:-max_keep]:
            ckpt.unlink(missing_ok=True)


# ============================================================
# VISUALIZATION
# ============================================================
def plot_training_curves(train_accuracies, train_losses, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_accuracies) + 1)
    ax1.plot(epochs, train_accuracies, 'b-o', linewidth=2, label='Train Accuracy')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.set_title('Training Accuracy')
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_ylim([0, 1.05])
    ax2.plot(epochs, train_losses, 'r-o', linewidth=2, label='Train Loss')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.set_title('Training Loss')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.suptitle('SPROUT Enhanced Training', fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training curves saved to {save_path}")


def plot_confusion_matrix(true_labels, pred_labels, class_names, save_path):
    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved to {save_path}")


def plot_shot_comparison(shot_results, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    shots = list(shot_results.keys())
    accuracies = [r[0] for r in shot_results.values()]
    stds = [r[1] for r in shot_results.values()]
    ax.errorbar(shots, accuracies, yerr=stds, fmt='bo-', linewidth=2, markersize=10, capsize=5)
    for shot, acc in zip(shots, accuracies):
        ax.text(shot, acc + 0.02, f'{acc:.3f}', ha='center', fontweight='bold')
    ax.set_xlabel('Number of Shots (K)'); ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs. Number of Shots'); ax.grid(True, alpha=0.3)
    ax.set_xticks(shots); ax.set_ylim(bottom=0, top=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Shot comparison saved to {save_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='SPROUT Enhanced Training')
    parser.add_argument('--data_dir', type=str, default=Config.DATA_DIR)
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR)
    parser.add_argument('--backbone', type=str, default=Config.BACKBONE,
                        choices=['resnet50', 'efficientnet_b0', 'efficientnet_v2_s',
                                 'efficientnet_v2_m', 'efficientnet_v2_l'])
    parser.add_argument('--n_way', type=int, default=Config.N_WAY)
    parser.add_argument('--k_shot', type=int, default=Config.K_SHOT)
    parser.add_argument('--num_epochs', type=int, default=Config.NUM_EPOCHS)
    parser.add_argument('--num_episodes', type=int, default=Config.NUM_EPISODES)
    parser.add_argument('--lr', type=float, default=Config.LR)
    parser.add_argument('--gpu', action='store_true', default=True)
    parser.add_argument('--no-gpu', dest='gpu', action='store_false')
    args = parser.parse_args()

    Config.DATA_DIR = args.data_dir
    Config.OUTPUT_DIR = args.output_dir
    Config.BACKBONE = args.backbone
    Config.N_WAY = args.n_way
    Config.K_SHOT = args.k_shot
    Config.NUM_EPOCHS = args.num_epochs
    Config.NUM_EPISODES = args.num_episodes
    Config.LR = args.lr

    set_seed(Config.SEED)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"\nLoading dataset from: {Config.DATA_DIR}")
    train_dataset = LeafDiseaseDataset(Config.DATA_DIR, transform=train_transform, split='train')

    test_dataset = None
    test_split_dir = Path(Config.DATA_DIR) / 'test'
    has_test_classes = test_split_dir.exists() and any(d.is_dir() for d in test_split_dir.iterdir())
    if has_test_classes:
        test_dataset = LeafDiseaseDataset(Config.DATA_DIR, transform=test_transform, split='test')
    else:
        print(f"  WARNING: No test class folders found in {test_split_dir}")
        print(f"  Evaluation will be skipped. Add test data later to enable evaluation.")

    classes = train_dataset.get_classes()
    num_classes = len(classes)
    print(f"\nClasses ({num_classes}): {classes}")

    model = SPROUT(
        backbone=Config.BACKBONE, embed_dim=Config.EMBED_DIM,
        hidden_dims=Config.HIDDEN_DIMS, num_refinement_steps=Config.NUM_REFINEMENT_STEPS,
        temperature=Config.TEMPERATURE, dropout_rate=Config.DROPOUT_RATE, num_heads=Config.NUM_HEADS
    ).to(device)

    criterion = SPROUTLoss(alpha=Config.ALPHA, beta=Config.BETA, gamma=Config.GAMMA,
                           margin=Config.MARGIN, label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)

    # FIX: Properly combine warmup + cosine annealing using SequentialLR
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=Config.WARMUP_EPOCHS
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[Config.WARMUP_EPOCHS]
    )

    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu',
                         enabled=Config.USE_AMP and torch.cuda.is_available())

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {Config.BACKBONE}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    config_dict = {k: v for k, v in vars(Config).items() if not k.startswith('_')}
    config_dict['device'] = str(device)
    config_dict['num_classes'] = num_classes
    config_dict['classes'] = classes
    config_dict['total_params'] = total_params
    with open(os.path.join(Config.OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"SPROUT Enhanced Training - {Config.NUM_EPOCHS} epochs, {Config.NUM_EPISODES} episodes/epoch")
    print(f"{'='*60}")
    train_accuracies, train_losses = [], []
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")

        epoch_acc, epoch_loss = train_one_epoch(
            model, criterion, optimizer, scaler, train_dataset, Config, device, epoch
        )

        scheduler.step()
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        train_accuracies.append(epoch_acc)
        train_losses.append(epoch_loss)

        print(f"  Acc: {epoch_acc:.4f} | Loss: {epoch_loss:.4f} | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
        if torch.cuda.is_available():
            print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            if (epoch + 1) % 10 == 0:
                torch.cuda.empty_cache()

        is_best = epoch_acc > best_acc
        if is_best:
            best_acc = epoch_acc
            print(f"  ** New best accuracy: {best_acc:.4f} **")

        save_checkpoint(model, optimizer, epoch, epoch_acc, epoch_loss,
                       Config.OUTPUT_DIR, is_best=is_best)

        if Config.SAVE_BEST_ONLY and is_best:
            cleanup_old_checkpoints(Config.OUTPUT_DIR, max_keep=Config.MAX_CHECKPOINTS)

        if (epoch + 1) % 5 == 0 or epoch == Config.NUM_EPOCHS - 1:
            plot_training_curves(train_accuracies, train_losses,
                                os.path.join(Config.OUTPUT_DIR, 'training_curves.png'))

    total_time = time.time() - start_time
    torch.save(model.state_dict(), os.path.join(Config.OUTPUT_DIR, 'sprout_final.pth'))
    plot_training_curves(train_accuracies, train_losses,
                        os.path.join(Config.OUTPUT_DIR, 'training_curves.png'))

    print(f"\n{'='*60}")
    print(f"Training Complete! Time: {total_time/60:.1f} minutes")
    print(f"Best accuracy: {best_acc:.4f}")

    # Multi-shot evaluation (only if test data exists)
    if test_dataset is not None and len(test_dataset) > 0:
        print(f"\n{'='*60}")
        print("Multi-Shot Evaluation")
        print(f"{'='*60}")

        best_model_path = os.path.join(Config.OUTPUT_DIR, 'sprout_best.pth')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

        shot_results = {}
        for shot in Config.TEST_SHOTS:
            acc, std, preds, labels = evaluate(model, test_dataset, Config, device, shot=shot)
            shot_results[shot] = (acc, std)
            print(f"  {shot}-shot: {acc:.4f} (+/- {std:.4f})")

        valid_shots = {k: v for k, v in shot_results.items() if v[0] > 0}
        if valid_shots:
            best_shot = max(valid_shots, key=lambda k: valid_shots[k][0])
            _, _, best_preds, best_labels = evaluate(model, test_dataset, Config, device, shot=best_shot)
            if best_labels:
                plot_confusion_matrix(best_labels, best_preds, classes,
                                     os.path.join(Config.OUTPUT_DIR, 'confusion_matrix.png'))
            plot_shot_comparison(shot_results, os.path.join(Config.OUTPUT_DIR, 'shot_comparison.png'))
    else:
        print("\nSkipping evaluation - no test data available.")
        shot_results = {}

    results = {
        'config': {k: v for k, v in vars(Config).items() if not k.startswith('_')},
        'shot_results': {str(k): {'accuracy': v[0], 'std': v[1]} for k, v in shot_results.items()},
        'best_accuracy': best_acc,
        'total_time_minutes': total_time / 60,
        'total_params': total_params,
        'num_classes': num_classes,
        'classes': classes
    }
    with open(os.path.join(Config.OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nAll files saved to: {Config.OUTPUT_DIR}")
    print("\nFiles:")
    for f_name in sorted(os.listdir(Config.OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(Config.OUTPUT_DIR, f_name))
        if size > 1024 * 1024:
            print(f"  {f_name} ({size/1024/1024:.1f} MB)")
        else:
            print(f"  {f_name} ({size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
