# SPROUT Training on Kaggle (with GPU)
# =====================================
# 1. Enable GPU in Kaggle Settings
# 2. Upload plantvillage dataset
# 3. Copy cells into a Kaggle notebook and run

# Cell 1: Install packages
# !pip install -q timm

# Cell 2: Import libraries
import os
import sys
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
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# Cell 3: Configuration
class Config:
    DATA_DIR = None  # Auto-detected
    BACKBONE = 'efficientnet_v2_s'
    EMBED_DIM = 128
    HIDDEN_DIMS = [512, 256]
    NUM_REFINEMENT_STEPS = 3
    TEMPERATURE = 10.0
    DROPOUT_RATE = 0.3
    NUM_HEADS = 4
    LABEL_SMOOTHING = 0.1
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
    USE_AMP = True
    SEED = 42
    OUTPUT_DIR = '/kaggle/working/sprout_results'
    NUM_TEST_EPISODES = 200
    TEST_SHOTS = [1, 3, 5, 10]

def detect_data_dir():
    possible_paths = [
        '/kaggle/input/plantvillage/plantvillage',
        '/kaggle/input/plantvillage',
        '/kaggle/input/plantvillage-dataset/PlantVillage',
        '/kaggle/input/plantvillage-dataset/plantvillage',
    ]
    for path in possible_paths:
        if os.path.exists(path):
            if os.path.exists(os.path.join(path, 'train')) and os.path.exists(os.path.join(path, 'test')):
                return path
            for sub in os.listdir(path):
                sub_path = os.path.join(path, sub)
                if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, 'train')):
                    return sub_path
    return None

Config.DATA_DIR = detect_data_dir()
if Config.DATA_DIR is None:
    print("ERROR: Could not find PlantVillage dataset! Add it as a Kaggle input.")
else:
    print(f"Dataset found at: {Config.DATA_DIR}")

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# Cell 4: Set seeds and device
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(Config.SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Cell 5: Dataset class
class LeafDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split_dir = self.root_dir / split
        self.classes = sorted([d.name for d in self.split_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []
        for class_name in self.classes:
            class_dir = self.split_dir / class_name
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                for img_path in class_dir.glob(f'**/{ext}'):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
        self.labels = torch.tensor(self.labels)
        self.indices_by_class = {}
        for class_idx in range(len(self.classes)):
            self.indices_by_class[class_idx] = torch.where(self.labels == class_idx)[0].tolist()
        print(f"  {split}: {len(self.image_paths)} images across {len(self.classes)} classes")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception:
            image = torch.zeros((3, 224, 224))
        return image, label

    def get_classes(self):
        return self.classes

# Cell 6: Model components
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
    def __init__(self, input_dim, embed_dim=128, hidden_dims=[512, 256], dropout_rate=0.3):
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
            prototype = torch.mean(class_embeddings, dim=0) if len(class_embeddings) > 0 else torch.zeros(support_embeddings.size(1), device=support_embeddings.device)
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
    def __init__(self, num_classes, backbone='resnet50', embed_dim=128,
                 hidden_dims=[512, 256], num_refinement_steps=3, temperature=10.0,
                 dropout_rate=0.3, num_heads=4):
        super().__init__()
        self.feature_extractor = FeatureExtractor(backbone=backbone)
        feature_dim = self.feature_extractor.get_feature_dim()
        self.embedding_network = EmbeddingNetwork(input_dim=feature_dim, embed_dim=embed_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate)
        self.prototype_module = PrototypeModule(embed_dim=embed_dim, num_refinement_steps=num_refinement_steps, num_heads=num_heads)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

    def forward(self, query_images, support_images=None, support_labels=None):
        query_features = self.feature_extractor(query_images)
        query_embeddings = self.embedding_network(query_features)
        if support_images is not None and support_labels is not None:
            support_features = self.feature_extractor(support_images)
            support_embeddings = self.embedding_network(support_features)
            prototypes = self.prototype_module(support_embeddings, support_labels)
            logits = -self.compute_distances(query_embeddings, prototypes)
            return logits, prototypes
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

# Cell 7: Loss function
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
        intra_loss = 0.0
        for c in classes:
            mask = (support_labels == c)
            if torch.sum(mask) > 1:
                class_emb = support_embeddings[mask]
                centroid = torch.mean(class_emb, dim=0, keepdim=True)
                intra_loss += torch.var(torch.sum((class_emb - centroid) ** 2, dim=1))
        intra_loss = self.beta * intra_loss / len(classes) if len(classes) > 0 else torch.tensor(0.0, device=logits.device)
        if prototypes.size(0) > 1:
            distances = torch.cdist(prototypes, prototypes, p=2)
            mask_upper = torch.triu(torch.ones_like(distances), diagonal=1) == 1
            inter_loss = self.gamma * F.relu(self.margin - distances[mask_upper]).mean()
        else:
            inter_loss = torch.tensor(0.0, device=logits.device)
        total_loss = proto_loss + refine_loss + intra_loss + inter_loss
        return total_loss, {
            'proto_loss': proto_loss.item(), 'refine_loss': refine_loss.item(),
            'intra_loss': intra_loss.item() if isinstance(intra_loss, torch.Tensor) else intra_loss,
            'inter_loss': inter_loss.item() if isinstance(inter_loss, torch.Tensor) else inter_loss,
            'total_loss': total_loss.item()
        }

# Cell 8: Episode creation
def create_episode(dataset, n_way, k_shot, n_query):
    indices_by_class = dataset.indices_by_class
    available_classes = [c for c in indices_by_class if len(indices_by_class[c]) >= k_shot + n_query]
    if len(available_classes) < n_way:
        available_classes = [c for c in indices_by_class if len(indices_by_class[c]) >= 2]
        n_way = min(n_way, len(available_classes))
    selected_classes = random.sample(available_classes, n_way)
    support_images, support_labels = [], []
    query_images, query_labels = [], []
    for new_label, class_idx in enumerate(selected_classes):
        indices = indices_by_class[class_idx]
        random.shuffle(indices)
        for idx in indices[:k_shot]:
            img, _ = dataset[idx]
            support_images.append(img)
            support_labels.append(new_label)
        for idx in indices[k_shot:k_shot + n_query]:
            img, _ = dataset[idx]
            query_images.append(img)
            query_labels.append(new_label)
    support_images = torch.stack(support_images).to(device)
    support_labels = torch.tensor(support_labels).to(device)
    query_images = torch.stack(query_images).to(device)
    query_labels = torch.tensor(query_labels).to(device)
    return support_images, support_labels, query_images, query_labels

# Cell 9: Load dataset
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

print("Loading dataset...")
train_dataset = LeafDiseaseDataset(Config.DATA_DIR, transform=train_transform, split='train')
test_dataset = LeafDiseaseDataset(Config.DATA_DIR, transform=test_transform, split='test')
classes = train_dataset.get_classes()
num_classes = len(classes)
print(f"\nClasses ({num_classes}): {classes}")

# Cell 10: Create model
model = SPROUT(
    num_classes=num_classes, backbone=Config.BACKBONE, embed_dim=Config.EMBED_DIM,
    hidden_dims=Config.HIDDEN_DIMS, num_refinement_steps=Config.NUM_REFINEMENT_STEPS,
    temperature=Config.TEMPERATURE, dropout_rate=Config.DROPOUT_RATE, num_heads=Config.NUM_HEADS
).to(device)

criterion = SPROUTLoss(alpha=Config.ALPHA, beta=Config.BETA, gamma=Config.GAMMA, margin=Config.MARGIN, label_smoothing=Config.LABEL_SMOOTHING)
optimizer = optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)

# Cosine annealing with warm restarts
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

# Warmup scheduler
warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=Config.WARMUP_EPOCHS)

scaler = GradScaler(enabled=Config.USE_AMP and torch.cuda.is_available())

print(f"Model: {Config.BACKBONE}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Mixed precision: {Config.USE_AMP and torch.cuda.is_available()}")

# Cell 11: Training
def train_one_epoch(model, criterion, optimizer, scaler, dataset, config):
    model.train()
    episode_accuracies, episode_losses = [], []
    loss_comp_sum = {'proto_loss': 0, 'refine_loss': 0, 'intra_loss': 0, 'inter_loss': 0, 'total_loss': 0}
    pbar = tqdm(range(config.NUM_EPISODES), desc="  Episodes", leave=False)
    for _ in pbar:
        optimizer.zero_grad()
        support_images, support_labels, query_images, query_labels = create_episode(dataset, config.N_WAY, config.K_SHOT, config.N_QUERY)
        with autocast(enabled=config.USE_AMP and torch.cuda.is_available()):
            logits, prototypes = model(query_images, support_images, support_labels)
            with torch.no_grad():
                support_features = model.feature_extractor(support_images)
                support_embeddings = model.embedding_network(support_features)
                initial_prototypes = model.prototype_module.generate_initial_prototypes(support_embeddings, support_labels)
            loss, loss_components = criterion(logits, query_labels, prototypes, initial_prototypes, support_embeddings, support_labels)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        _, predicted = torch.max(logits.data, 1)
        accuracy = (predicted == query_labels).float().mean().item()
        episode_accuracies.append(accuracy)
        episode_losses.append(loss_components['total_loss'])
        for k in loss_comp_sum:
            loss_comp_sum[k] += loss_components[k]
        pbar.set_postfix(acc=f"{accuracy:.3f}", loss=f"{loss_components['total_loss']:.4f}")
    n = config.NUM_EPISODES
    return np.mean(episode_accuracies), np.mean(episode_losses), {k: v / n for k, v in loss_comp_sum.items()}

print("Starting SPROUT Training...")
print("=" * 60)
train_accuracies, train_losses = [], []
best_acc = 0.0
start_time = time.time()

for epoch in range(Config.NUM_EPOCHS):
    epoch_start = time.time()
    print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
    print("-" * 40)
    epoch_acc, epoch_loss, loss_comp = train_one_epoch(model, criterion, optimizer, scaler, train_dataset, Config)

    # Update learning rate with warmup
    if epoch < Config.WARMUP_EPOCHS:
        warmup_scheduler.step()
    else:
        scheduler.step()

    epoch_time = time.time() - epoch_start
    train_accuracies.append(epoch_acc)
    train_losses.append(epoch_loss)
    print(f"  Accuracy: {epoch_acc:.4f} | Loss: {epoch_loss:.4f} | Time: {epoch_time:.1f}s")
    if torch.cuda.is_available():
        print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    if (epoch + 1) % 5 == 0 or epoch == Config.NUM_EPOCHS - 1:
        torch.save(model.state_dict(), os.path.join(Config.OUTPUT_DIR, f'sprout_epoch_{epoch+1}.pth'))
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(model.state_dict(), os.path.join(Config.OUTPUT_DIR, 'sprout_best.pth'))

total_time = time.time() - start_time
torch.save(model.state_dict(), os.path.join(Config.OUTPUT_DIR, 'sprout_final.pth'))
print(f"\nTraining Complete! Total time: {total_time/60:.1f} minutes")
print(f"Best accuracy: {best_acc:.4f}")

# Cell 12: Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(range(1, Config.NUM_EPOCHS + 1), train_accuracies, 'b-o', linewidth=2, label='Train Accuracy')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.set_title('Training Accuracy'); ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_ylim([0, 1.05])
ax2.plot(range(1, Config.NUM_EPOCHS + 1), train_losses, 'r-o', linewidth=2, label='Train Loss')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.set_title('Training Loss'); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.suptitle(f'SPROUT Training ({Config.BACKBONE})', fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.show()

# Cell 13: Evaluate
model.load_state_dict(torch.load(os.path.join(Config.OUTPUT_DIR, 'sprout_best.pth'), map_location=device))
model.eval()
all_preds, all_labels, episode_accs = [], [], []
for _ in tqdm(range(Config.NUM_TEST_EPISODES), desc="Evaluating"):
    support_images, support_labels, query_images, query_labels = create_episode(test_dataset, Config.N_WAY, Config.K_SHOT, Config.N_QUERY)
    with torch.no_grad():
        logits, _ = model(query_images, support_images, support_labels)
    _, predicted = torch.max(logits.data, 1)
    episode_accs.append((predicted == query_labels).float().mean().item())
    all_preds.extend(predicted.cpu().numpy())
    all_labels.extend(query_labels.cpu().numpy())

print(f"\nTest Accuracy: {np.mean(episode_accs):.4f} (+/- {np.std(episode_accs):.4f})")
print(classification_report(all_labels, all_preds, target_names=test_dataset.get_classes(), zero_division=0))

# Cell 14: Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()

# Cell 15: Save results
results = {
    'config': {'backbone': Config.BACKBONE, 'n_way': Config.N_WAY, 'k_shot': Config.K_SHOT, 'lr': Config.LR},
    'test_accuracy': float(np.mean(episode_accs)),
    'test_std': float(np.std(episode_accs)),
    'best_train_accuracy': float(best_acc),
    'total_time_minutes': total_time / 60,
    'num_classes': num_classes,
    'classes': test_dataset.get_classes()
}
with open(os.path.join(Config.OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\nFiles saved:")
for f in sorted(os.listdir(Config.OUTPUT_DIR)):
    print(f"  {f}")
print(f"\nDownload from: {Config.OUTPUT_DIR}")
