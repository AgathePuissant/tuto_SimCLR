import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
import os
import copy
import time
import random
import logging
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from datetime import datetime
from typing import Optional, Tuple, Union
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
from sklearn.neighbors import NearestNeighbors
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.caching").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Backbone registry
# ---------------------------------------------------------------------------

_BACKBONE_MAP = {
    "resnet18":  models.resnet18,
    "resnet34":  models.resnet34,
    "resnet50":  models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}

_BACKBONE_FEAT_DIM = {
    "resnet18": 512, "resnet34": 512,
    "resnet50": 2048, "resnet101": 2048, "resnet152": 2048,
}

IMAGENET_NORMALIZE = {
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225],
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """ResNet backbone with a two-layer projection head.

    Checkpoint format:
        {"model_state", "backbone", "out_dim", "saved_at"}
    """

    def __init__(self, backbone: str = "resnet50", out_dim: int = 128, pretrained: bool = True):
        super().__init__()
        if backbone not in _BACKBONE_MAP:
            raise ValueError(f"Unknown backbone '{backbone}'. Available: {list(_BACKBONE_MAP.keys())}")

        self.backbone_name = backbone
        self.out_dim = out_dim
        self.feature_dim = _BACKBONE_FEAT_DIM[backbone]

        weights = "DEFAULT" if pretrained else None
        self.encoder = _BACKBONE_MAP[backbone](weights=weights)
        self.encoder.fc = nn.Identity()

        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        return self.projection_head(self.encoder(x))

    def encode_backbone(self, x):
        return self.encoder(x)


def save_model(model: Encoder, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "backbone":    model.backbone_name,
            "out_dim":     model.out_dim,
            "saved_at":    datetime.now().isoformat(),
        },
        path,
    )


def save_full_checkpoint(model: Encoder, optimizer, epoch: int, save_path: str, hparams: dict):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "epoch":           epoch,
            "hparams":         hparams,
            "saved_at":        datetime.now().isoformat(),
        },
        save_path,
    )


def load_model(path: str, device="cpu") -> Encoder:
    """Load an Encoder from a .pth file.

    Handles both the enriched dict format and raw state-dicts from older runs.
    """
    raw = torch.load(path, map_location=device)
    if isinstance(raw, dict) and "model_state" in raw:
        model = Encoder(
            backbone=raw.get("backbone", "resnet50"),
            out_dim=raw.get("out_dim", 128),
            pretrained=False,
        )
        model.load_state_dict(raw["model_state"])
    else:
        model = Encoder(backbone="resnet50", out_dim=128, pretrained=False)
        model.load_state_dict(raw)
    return model


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def nt_xent_loss_multi(z, batch_size, num_views, temperature=0.5):
    """NT-Xent loss for an arbitrary number of views.

    Args:
        z: (num_views * B, D) tensor of L2-normalised projections.
        batch_size: B
        num_views: N
    """
    device = z.device
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.T) / temperature

    diag_mask = torch.eye(sim.size(0), dtype=torch.bool, device=device)
    sim_exp = torch.exp(sim.masked_fill(diag_mask, -9e15))

    sample_ids = torch.arange(sim.size(0), device=device) // num_views
    positive_mask = (sample_ids.unsqueeze(0) == sample_ids.unsqueeze(1)) & ~diag_mask

    denom   = sim_exp.sum(dim=1)
    pos_exp = (sim_exp * positive_mask.float()).sum(dim=1)

    loss = -torch.log((pos_exp + 1e-8) / (denom + 1e-8))
    return loss.mean()


class SupConLossLite(nn.Module):
    """Supervised contrastive loss supporting multi-view batches.

    Two modes:
    - use_all_pairs=True  : full (V*B)² mask, exact but memory-heavy.
    - use_all_pairs=False : stochastic sampling, memory-friendly fallback.
    """

    def __init__(self, temperature=0.07, use_all_pairs=True, samples_per_image=2):
        super().__init__()
        self.temperature = temperature
        self.use_all_pairs = use_all_pairs
        self.samples_per_image = samples_per_image

    def forward(self, features, labels):
        """
        Args:
            features: list of V tensors each [B, D]
            labels:   [B] long tensor
        """
        device = features[0].device
        labels = labels.to(device)
        B = labels.shape[0]
        V = len(features)

        feats = F.normalize(torch.cat(features, dim=0), dim=1)

        if self.use_all_pairs:
            mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)
            pos_mask = mask.repeat(V, V)
            self_mask = torch.eye(V * B, device=device).bool()
            pos_mask = pos_mask.masked_fill(self_mask, 0)

            sim = torch.matmul(feats, feats.T) / self.temperature
            exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
            denom = exp_sim.sum(dim=1)
            pos_sim = (exp_sim * pos_mask).sum(dim=1)
            loss = -torch.log((pos_sim + 1e-12) / (denom + 1e-12))
            return loss.mean()

        views = F.normalize(torch.stack(features, dim=0), dim=-1)
        all_embs = views.reshape(-1, views.shape[-1])

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_pairs = 0
        for b in range(B):
            same_idx = (labels == labels[b]).nonzero(as_tuple=False).squeeze(1)
            if same_idx.numel() <= 1:
                same_idx = torch.tensor([i for i in range(B) if i != b], device=device)
                if same_idx.numel() == 0:
                    continue
            for _ in range(self.samples_per_image):
                i = torch.randint(0, V, (1,)).item()
                p_sample = same_idx[torch.randint(0, same_idx.numel(), (1,)).item()].item()
                j = torch.randint(0, V, (1,)).item()
                zi = views[i, b]
                zj = views[j, p_sample]
                pos_sim = torch.matmul(zi, zj) / self.temperature
                logits = torch.matmul(zi, all_embs.T) / self.temperature
                pos_index = i * B + b
                logits = torch.cat([logits[:pos_index], logits[pos_index + 1:]])
                total_loss = total_loss + (-pos_sim + torch.logsumexp(logits, dim=0))
                total_pairs += 1

        if total_pairs == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return total_loss / total_pairs


# ---------------------------------------------------------------------------
# Data augmentation & collation
# ---------------------------------------------------------------------------

class SimCLRTransform:
    """Stochastic augmentation pipeline returning N independent views of an image."""

    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 0.5,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[int] = None,
        sigmas: Tuple[float, float] = (0.2, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        normalize: dict = IMAGENET_NORMALIZE,
        num_views: int = 2,
    ):
        if kernel_size is None:
            kernel_size = int(0.1 * input_size)
            if kernel_size % 2 == 0:
                kernel_size += 1

        transform_list = [
            transforms.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
        ]

        if rr_prob > 0 and rr_degrees is not None:
            transform_list.append(
                transforms.RandomApply([transforms.RandomRotation(rr_degrees)], p=rr_prob)
            )
        if hf_prob > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=hf_prob))
        if vf_prob > 0:
            transform_list.append(transforms.RandomVerticalFlip(p=vf_prob))
        if cj_prob > 0:
            transform_list.append(
                transforms.RandomApply(
                    [transforms.ColorJitter(
                        brightness=cj_strength,
                        contrast=cj_strength,
                        saturation=cj_strength,
                        hue=0.1 * cj_strength,
                    )],
                    p=cj_prob,
                )
            )
        if random_gray_scale > 0:
            transform_list.append(transforms.RandomGrayscale(p=random_gray_scale))
        if gaussian_blur > 0:
            transform_list.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigmas)],
                    p=gaussian_blur,
                )
            )

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(**normalize),
        ])

        self.transform = transforms.Compose(transform_list)
        self.num_views = int(num_views)

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.num_views)]


def simclr_collate(batch):
    """Collate function for multi-view batches.

    Args:
        batch: list of (views_list, label) where views_list = [v1, ..., vN], each [C, H, W]
    Returns:
        views:  list of N tensors each [B, C, H, W]
        labels: [B] long tensor
    """
    views_per_view = list(zip(*[item[0] for item in batch]))
    views = [torch.stack(vs, dim=0) for vs in views_per_view]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return views, labels


def display_augmentations(dataset_path, transform, n=8):
    all_imgs = [
        os.path.join(root, f)
        for root, _, files in os.walk(dataset_path)
        for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not all_imgs:
        st.warning("No images found in dataset folder.")
        return

    img = Image.open(random.choice(all_imgs)).convert("RGB")
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
    for i in range(n):
        views = transform(img)
        tensor = views[i % len(views)]
        arr = tensor.permute(1, 2, 0).numpy()
        arr = arr * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        axes[i].imshow(np.clip(arr, 0, 1))
        axes[i].axis("off")
    plt.tight_layout()
    st.pyplot(fig)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def knn_cosine_accuracy(train_feats, train_labels, val_feats, val_labels, k=1, device="cuda"):
    """k-NN accuracy using cosine similarity, computed on GPU when available."""
    train_feats  = F.normalize(train_feats.to(device),  dim=1)
    val_feats    = F.normalize(val_feats.to(device),    dim=1)
    train_labels = train_labels.to(device)
    val_labels   = val_labels.to(device)

    sims     = torch.mm(val_feats, train_feats.T)
    topk_idx = sims.topk(k, dim=1).indices
    topk_labels = train_labels[topk_idx]

    preds = topk_labels.squeeze(1) if k == 1 else torch.mode(topk_labels, dim=1).values
    return (preds == val_labels).float().mean().item()


def _extract_knn_splits(labeled_path, transform_eval, device):
    """Helper: extract train/val embeddings from an ImageFolder for k-NN eval."""
    dataset = datasets.ImageFolder(root=labeled_path, transform=transform_eval)
    if len(dataset) < 2:
        return None, None, None, None

    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(0.8 * n)

    train_loader = DataLoader(Subset(dataset, indices[:split]), batch_size=64, shuffle=False, num_workers=4)
    val_loader   = DataLoader(Subset(dataset, indices[split:]), batch_size=64, shuffle=False, num_workers=4)

    def _collect(loader, model):
        feats_list, labels_list = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                feats_list.append(model.encode_backbone(imgs.to(device)).cpu().numpy())
                labels_list.extend(lbls.numpy().tolist())
        return np.vstack(feats_list), labels_list

    train_f, train_l = _collect(train_loader, None)
    return train_f, train_l, *_collect(val_loader, None)


def _build_eval_transform(train_transform, fallback_size, normalize):
    if hasattr(train_transform.transform.transforms[0], "size"):
        sz = train_transform.transform.transforms[0].size
        sz = (sz, sz) if isinstance(sz, int) else sz
    else:
        sz = (fallback_size, fallback_size)
    return transforms.Compose([
        transforms.Resize(sz),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize["mean"], std=normalize["std"]),
    ])


def _run_knn_eval(model, transform, labeled_path, fallback_size, normalize, knn_k, device):
    """Run k-NN evaluation and display results in the Streamlit UI."""
    st.write("Running k-NN evaluation on labeled dataset...")
    try:
        eval_transform = _build_eval_transform(transform, fallback_size, normalize)
        dataset = datasets.ImageFolder(root=labeled_path, transform=eval_transform)
        if len(dataset) < 2:
            st.warning("Labeled dataset needs at least 2 images.")
            return None

        n = len(dataset)
        indices = list(range(n))
        random.shuffle(indices)
        split = int(0.8 * n)

        def _collect(loader):
            feats_list, labels_list = [], []
            with torch.no_grad():
                for imgs, lbls in loader:
                    feats_list.append(model.encode_backbone(imgs.to(device)).cpu().numpy())
                    labels_list.extend(lbls.numpy().tolist())
            return feats_list, labels_list

        model.eval()
        train_f, train_l = _collect(DataLoader(Subset(dataset, indices[:split]), batch_size=64, num_workers=4))
        val_f,   val_l   = _collect(DataLoader(Subset(dataset, indices[split:]), batch_size=64, num_workers=4))

        if not train_f or not val_f:
            st.warning("k-NN evaluation skipped: empty train/val splits.")
            return None

        tf = torch.tensor(np.vstack(train_f), dtype=torch.float32)
        vf = torch.tensor(np.vstack(val_f),   dtype=torch.float32)
        tl = torch.tensor(train_l, dtype=torch.long)
        vl = torch.tensor(val_l,   dtype=torch.long)

        acc = knn_cosine_accuracy(tf, tl, vf, vl, k=knn_k, device=device)
        st.write(f"k-NN accuracy (k={knn_k}): {acc * 100:.2f}%")
        return acc

    except Exception as e:
        st.write(f"k-NN evaluation failed: {e}")
        return None


def _log_epoch(log_csv_path, log_columns, row_dict):
    try:
        prev = pd.read_csv(log_csv_path, sep=";")
    except Exception:
        prev = pd.DataFrame(columns=log_columns)
    prev = pd.concat([prev, pd.DataFrame([row_dict])], ignore_index=True)
    prev.to_csv(log_csv_path, index=False, sep=";")


# ---------------------------------------------------------------------------
# Generic contrastive training loop (shared by SimCLR and SupCon)
# ---------------------------------------------------------------------------

def _train_contrastive(
    model,
    dataloader,
    criterion,
    epochs,
    lr,
    device,
    transform,
    save_model_path,
    normalize,
    checkpoint_freq=10,
    enable_knn=False,
    labeled_dataset_path=None,
    knn_k=1,
    save_best_model=True,
    augment_params=None,
    dataset_name=None,
    run_label="SimCLR",
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    loss_history = []
    plot_spot  = st.empty()
    logtxtbox  = st.empty()

    fig, ax = plt.subplots()
    loss_line, = ax.plot([], [], marker="o", label="Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{run_label} Loss")
    ax.legend()

    model_dir      = os.path.dirname(save_model_path) or "."
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(save_model_path))[0]
    log_csv_path = os.path.join(model_dir, f"{stem}_training_log.csv")
    log_columns  = ["epoch", "loss", "knn_acc", "lr", "batch_size", "dataset_name", "augment_params", "timestamp"]
    pd.DataFrame(columns=log_columns).to_csv(log_csv_path, index=False, sep=";")

    best_metric = -1.0
    best_loss   = float("inf")
    best_epoch  = -1
    input_size  = (
        augment_params.get("input_size", 224)
        if isinstance(augment_params, dict)
        else 224
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

        for views, labels in progress_bar:
            views  = [v.to(device) for v in views]
            labels = labels.to(device)

            if isinstance(criterion, SupConLossLite):
                feats = [model(v) for v in views]
                loss  = criterion(feats, labels)
            else:
                z    = torch.cat([model(v) for v in views], dim=0)
                loss = criterion(z, batch_size=views[0].size(0), num_views=len(views))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        loss_line.set_data(range(1, len(loss_history) + 1), loss_history)
        ax.relim()
        ax.autoscale_view()

        logtxtbox.write(f"Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")
        with plot_spot:
            st.pyplot(fig)

        knn_acc = None
        if (epoch + 1) % checkpoint_freq == 0 or (epoch + 1) == epochs:
            chk_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            save_full_checkpoint(model, optimizer, epoch + 1, chk_path, hparams=augment_params or {})
            st.write(f"Checkpoint saved: {chk_path}")

            if enable_knn and labeled_dataset_path and os.path.isdir(labeled_dataset_path):
                knn_acc = _run_knn_eval(
                    model, transform, labeled_dataset_path, input_size, normalize, knn_k, device
                )

            if enable_knn and knn_acc is not None:
                if knn_acc > best_metric:
                    best_metric = knn_acc
                    best_epoch  = epoch + 1
                    if save_best_model:
                        best_path = os.path.join(model_dir, "best_model.pth")
                        save_model(model, best_path)
                        st.write(f"Best model updated (k-NN acc={best_metric:.4f}) → {best_path}")
            else:
                if avg_loss < best_loss:
                    best_loss  = avg_loss
                    best_epoch = epoch + 1
                    if save_best_model:
                        best_path = os.path.join(model_dir, "best_model.pth")
                        save_model(model, best_path)
                        st.write(f"Best model updated (loss={best_loss:.6f}) → {best_path}")

        _log_epoch(log_csv_path, log_columns, {
            "epoch":         epoch + 1,
            "loss":          avg_loss,
            "knn_acc":       knn_acc if knn_acc is not None else "",
            "lr":            lr,
            "batch_size":    dataloader.batch_size,
            "dataset_name":  dataset_name or stem,
            "augment_params": str(augment_params) if augment_params else "",
            "timestamp":     datetime.now().isoformat(),
        })

    final_path = os.path.join(model_dir, f"{stem}_last.pth")
    save_model(model, final_path)
    st.success(f"Final model saved → {final_path}")
    best_val = best_metric if enable_knn else best_loss
    st.info(f"Best epoch: {best_epoch}  (metric={best_val})")
    return model


def _build_weighted_dataloader(dataset, batch_size, collate_fn=None):
    class_counts = [0] * len(dataset.classes)
    for _, label in dataset.samples:
        class_counts[label] += 1
    class_weights = [1.0 / c if c > 0 else 0.0 for c in class_counts]
    weights = [class_weights[label] for _, label in dataset.samples]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        collate_fn=collate_fn,
    )


def _augmentation_sidebar(key_suffix=""):
    """Render shared augmentation controls and return a params dict."""
    input_size = st.number_input("Input size", 64, 512, 224, 8, key=f"input_size{key_suffix}")
    min_scale  = st.slider("Min scale (RandomResizedCrop)", 0.01, 1.0, 0.3, 0.01, key=f"min_scale{key_suffix}")

    col1, col2 = st.columns(2)
    with col1:
        cj_prob = st.slider("Color jitter probability", 0.0, 1.0, 0.5, 0.01, key=f"cj_prob{key_suffix}")
    with col2:
        cj_strength = st.slider("Color jitter strength", 0.0, 1.0, 0.3, 0.01, key=f"cj_str{key_suffix}")

    col3, col4 = st.columns(2)
    with col3:
        hf_prob = st.slider("Horizontal flip prob.", 0.0, 1.0, 0.5, 0.01, key=f"hf{key_suffix}")
    with col4:
        vf_prob = st.slider("Vertical flip prob.", 0.0, 1.0, 0.5, 0.01, key=f"vf{key_suffix}")

    col5, col6 = st.columns(2)
    with col5:
        gray_prob = st.slider("Grayscale prob.", 0.0, 1.0, 0.3, 0.01, key=f"gray{key_suffix}")
    with col6:
        blur_prob = st.slider("Gaussian blur prob.", 0.0, 1.0, 0.3, 0.01, key=f"blur{key_suffix}")

    col7, col8 = st.columns(2)
    with col7:
        sigmas_min = st.number_input("Min sigma (blur)", 0.01, 5.0, 0.2, 0.01, key=f"smin{key_suffix}")
    with col8:
        sigmas_max = st.number_input("Max sigma (blur)", 0.1, 5.0, 2.0, 0.01, key=f"smax{key_suffix}")

    rotation   = st.checkbox("Enable random rotation", value=False, key=f"rot{key_suffix}")
    rr_prob    = st.slider("Rotation prob.", 0.0, 1.0, 0.5, 0.01, key=f"rrp{key_suffix}") if rotation else 0.0
    rr_degrees = st.slider("Rotation range (°)", 0, 180, 45, 1, key=f"rrd{key_suffix}") if rotation else None

    return dict(
        input_size=input_size, min_scale=min_scale,
        cj_prob=cj_prob, cj_strength=cj_strength,
        hf_prob=hf_prob, vf_prob=vf_prob,
        gray_prob=gray_prob, blur_prob=blur_prob,
        sigmas_min=sigmas_min, sigmas_max=sigmas_max,
        rr_prob=rr_prob, rr_degrees=rr_degrees,
    )


def _make_transform(p, normalize, num_views):
    return SimCLRTransform(
        input_size=p["input_size"],
        cj_prob=p["cj_prob"],
        cj_strength=p["cj_strength"],
        min_scale=p["min_scale"],
        random_gray_scale=p["gray_prob"],
        gaussian_blur=p["blur_prob"],
        sigmas=(p["sigmas_min"], p["sigmas_max"]),
        vf_prob=p["vf_prob"],
        hf_prob=p["hf_prob"],
        rr_prob=p["rr_prob"],
        rr_degrees=(-p["rr_degrees"], p["rr_degrees"]) if p["rr_degrees"] else None,
        normalize=normalize,
        num_views=num_views,
    )


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

st.title("Contrastive learning GUI")

page = st.sidebar.radio(
    "Navigation",
    ["SimCLR training", "SupCon Training", "Generate Embeddings", "Validation", "GradCAM", "Visualization"],
)


# ===========================================================================
# Training page
# ===========================================================================

if page == "SimCLR training":
    st.header("SimCLR Self-Supervised Training")

    dataset_path    = st.text_input("Dataset path", value="data")
    save_model_path = st.text_input("Model save path (.pth)", value="runs/simclr_model.pth")
    batch_size      = st.slider("Batch size", 16, 256, 128)
    epochs          = st.slider("Epochs", 1, 300, 300)
    learning_rate   = st.number_input("Learning rate", 1e-6, 1e-1, 1e-3, format="%.6f")

    st.subheader("Checkpointing & evaluation")
    checkpoint_freq    = st.number_input("Checkpoint every N epochs", 1, 100, 10, step=1)
    enable_knn         = st.checkbox("Enable k-NN evaluation at checkpoints", value=False)
    labeled_path       = st.text_input("Labeled dataset path (for k-NN)", value="") if enable_knn else ""
    knn_k              = st.slider("k (k-NN majority vote)", 1, 15, 5)
    save_best_model    = st.checkbox("Save best model", value=False)

    st.subheader("Architecture")
    backbone_simclr = st.selectbox("Backbone", list(_BACKBONE_MAP.keys()), index=2)
    out_dim_simclr  = st.number_input("Projection head output dim", 32, 512, 128, step=32)
    num_views       = st.slider("Number of views per image", 2, 6, 2)
    temperature     = st.number_input("NT-Xent temperature", 0.01, 1.0, 0.8, step=0.01, format="%.2f")

    st.subheader("Augmentations")
    aug_p = _augmentation_sidebar(key_suffix="_simclr")

    show_aug      = st.button("Preview augmentations")
    start_training = st.button("Start training")

    normalize = IMAGENET_NORMALIZE

    if show_aug and dataset_path:
        try:
            display_augmentations(dataset_path, _make_transform(aug_p, normalize, num_views), n=8)
        except Exception as e:
            st.error(f"Could not display augmentations: {e}")

    if start_training and dataset_path:
        try:
            
            st.write("Loading dataset…")
            transform = _make_transform(aug_p, normalize, num_views)
            dataset   = datasets.ImageFolder(root=dataset_path, transform=transform)
            dataloader = _build_weighted_dataloader(dataset, batch_size, collate_fn=simclr_collate)
            st.write(f"{len(dataset)} images loaded.")

            model  = Encoder(backbone=backbone_simclr, out_dim=int(out_dim_simclr))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            st.write(f"using device: {device}")

            def _criterion(z, batch_size, num_views):
                return nt_xent_loss_multi(z, batch_size, num_views, temperature=temperature)

            _train_contrastive(
                model=model,
                dataloader=dataloader,
                criterion=_criterion,
                epochs=epochs,
                lr=learning_rate,
                device=device,
                transform=transform,
                save_model_path=save_model_path,
                normalize=normalize,
                checkpoint_freq=int(checkpoint_freq),
                enable_knn=enable_knn,
                labeled_dataset_path=labeled_path,
                knn_k=int(knn_k),
                save_best_model=save_best_model,
                augment_params={**aug_p, "num_views": num_views, "temperature": temperature},
                dataset_name=os.path.basename(dataset_path.rstrip("/")),
                run_label="SimCLR",
            )
        except Exception as e:
            st.error(f"Training failed: {e}")


# ===========================================================================
# SupCon Training page
# ===========================================================================

elif page == "SupCon Training":
    st.header("Supervised Contrastive (SupCon) Training")

    dataset_path_sup    = st.text_input("Dataset path (subfolders = classes)", value="")
    save_model_path_sup = st.text_input("Model save path (.pth)", value="supcon_model.pth")
    batch_size_sup      = st.slider("Batch size", 8, 256, 32)
    epochs_sup          = st.slider("Epochs", 1, 300, 20)
    learning_rate_sup   = st.number_input("Learning rate", 1e-6, 1e-1, 1e-3, format="%.6f")

    st.subheader("Checkpointing & evaluation")
    checkpoint_freq_sup  = st.number_input("Checkpoint every N epochs", 1, 100, 10, step=1)
    enable_knn_sup       = st.checkbox("Enable k-NN evaluation at checkpoints", value=False)
    knn_k_sup            = st.slider("k (k-NN majority vote)", 1, 15, 1)
    save_best_model_sup  = st.checkbox("Save best model", value=True)

    st.subheader("Architecture")
    backbone_sup     = st.selectbox("Backbone", list(_BACKBONE_MAP.keys()), index=2)
    out_dim_sup      = st.number_input("Projection head output dim", 32, 512, 128, step=32)
    num_views_sup    = st.slider("Number of views per image", 2, 6, 2)
    temperature_sup  = st.number_input("SupCon temperature τ", 0.01, 1.0, 0.8, step=0.01, format="%.2f")
    samples_per_image_sup = st.number_input(
        "Samples per image (stochastic mode)", min_value=1, max_value=10, value=2, step=1
    )

    st.subheader("Augmentations")
    aug_p_sup = _augmentation_sidebar(key_suffix="_supcon")

    show_aug_sup  = st.button("Preview augmentations")
    start_supcon  = st.button("Start SupCon training")

    normalize_sup = IMAGENET_NORMALIZE

    if show_aug_sup and dataset_path_sup:
        try:
            display_augmentations(dataset_path_sup, _make_transform(aug_p_sup, normalize_sup, num_views_sup), n=8)
        except Exception as e:
            st.error(f"Could not display augmentations: {e}")

    if start_supcon and dataset_path_sup:
        try:
            st.write("Loading dataset…")
            transform_sup = _make_transform(aug_p_sup, normalize_sup, num_views_sup)
            dataset_sup   = datasets.ImageFolder(root=dataset_path_sup, transform=transform_sup)
            if len(dataset_sup) == 0:
                st.error("No images found in dataset path.")
            else:
                dataloader_sup = _build_weighted_dataloader(dataset_sup, batch_size_sup, collate_fn=simclr_collate)
                st.write(f"{len(dataset_sup)} images, {len(dataset_sup.classes)} classes loaded.")

                model_sup  = Encoder(backbone=backbone_sup, out_dim=out_dim_sup)
                device_sup = "cuda" if torch.cuda.is_available() else "cpu"
                st.write(f"using device: {device_sup}")
                
                use_all_pairs = (num_views_sup * batch_size_sup) <= 512 #Else memory overload
                criterion_sup = SupConLossLite(
                    temperature=temperature_sup,
                    use_all_pairs=use_all_pairs,
                    samples_per_image=samples_per_image_sup,
                )

                _train_contrastive(
                    model=model_sup,
                    dataloader=dataloader_sup,
                    criterion=criterion_sup,
                    epochs=epochs_sup,
                    lr=learning_rate_sup,
                    device=device_sup,
                    transform=transform_sup,
                    save_model_path=save_model_path_sup,
                    normalize=normalize_sup,
                    checkpoint_freq=int(checkpoint_freq_sup),
                    enable_knn=enable_knn_sup,
                    labeled_dataset_path=dataset_path_sup,
                    knn_k=int(knn_k_sup),
                    save_best_model=save_best_model_sup,
                    augment_params={**aug_p_sup, "num_views": num_views_sup, "temperature": temperature_sup},
                    dataset_name=os.path.basename(dataset_path_sup.rstrip("/")),
                    run_label="SupCon",
                )
        except Exception as e:
            st.error(f"SupCon training failed: {e}")


# ===========================================================================
# Generate Embeddings page
# ===========================================================================

elif page == "Generate Embeddings":
    st.header("Generate Embeddings")

    embeddings_folder = st.text_input("Image folder", value="")
    model_selection   = st.file_uploader("Model file (.pth)", type=["pth"])
    embedding_save_path = st.text_input("Output CSV path", value="embeddings.csv")

    image_paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(embeddings_folder)
        for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    start_embedding = st.button("Generate Embeddings")

    if start_embedding and embeddings_folder and model_selection:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
                tmp.write(model_selection.read())
                tmp_path = tmp.name
            model = load_model(tmp_path, device=str(device))
            os.unlink(tmp_path)
            model.eval().to(device)

            transform_e = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(**IMAGENET_NORMALIZE),
            ])

            class ImageDataset(Dataset):
                def __init__(self, paths, transform):
                    self.paths, self.transform = paths, transform

                def __len__(self):
                    return len(self.paths)

                def __getitem__(self, idx):
                    img = Image.open(self.paths[idx]).convert("RGB")
                    return self.transform(img), os.path.basename(self.paths[idx])

            dataloader = DataLoader(
                ImageDataset(image_paths, transform_e),
                batch_size=8, shuffle=False, num_workers=0, pin_memory=True,
            )

            st.write(f"Processing {len(image_paths)} images…")
            embeddings   = []
            progress_bar = st.progress(0)

            with torch.no_grad():
                for batch_idx, (images, names) in enumerate(dataloader):
                    batch_embs = model.encode_backbone(images.to(device)).cpu().numpy()
                    for name, emb in zip(names, batch_embs):
                        embeddings.append([name] + emb.tolist())
                    progress_bar.progress((batch_idx + 1) / len(dataloader))

            pd.DataFrame(embeddings).to_csv(embedding_save_path, index=False, sep=";")

            feats_array  = np.array([r[1:] for r in embeddings], dtype=np.float32)
            labels_array = np.array([os.path.basename(os.path.dirname(p)) for p in image_paths])
            out_dir      = os.path.dirname(embedding_save_path) or "."
            np.save(os.path.join(out_dir, "ref_feats.npy"),  feats_array)
            np.save(os.path.join(out_dir, "ref_labels.npy"), labels_array)

            st.success(f"Embeddings saved → {embedding_save_path}")
            st.success(f"ref_feats.npy  ({feats_array.shape[0]} × {feats_array.shape[1]}) saved.")
            st.success(f"ref_labels.npy ({len(labels_array)} labels) saved.")
            progress_bar.empty()

        except Exception as e:
            st.error(f"Error: {e}")


# ===========================================================================
# Validation page (linear probe)
# ===========================================================================

elif page == "Validation":
    st.header("Linear Probe Validation")

    model_selection   = st.file_uploader("Model file (.pth)", type=["pth"])
    dataset_path          = st.text_input("Validation dataset path", value="")
    batch_size_validation = st.slider("Batch size", 16, 128, 32)
    start_validation      = st.button("Start Validation")

    def train_linear_classifier(model, dataloaders, dataset_sizes, num_epochs=50, lr=0.001):
        since     = time.time()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.projection_head.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(num_epochs * 0.6), int(num_epochs * 0.8)],
            gamma=0.1,
        )
        best_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        logtxtbox  = st.empty()
        logtxtbox2 = st.empty()
        plot_spot  = st.empty()

        train_hist, val_hist = [], []
        fig, ax = plt.subplots()
        tl, = ax.plot([], [], marker="o", label="Train loss")
        vl, = ax.plot([], [], marker="x", label="Val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Linear Probe Training")
        ax.legend()

        for epoch in range(num_epochs):
            for phase in ["train", "validation"]:
                model.train() if phase == "train" else model.eval()
                running_loss, running_corrects = 0.0, 0

                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                    running_loss     += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc  = running_corrects.double() / dataset_sizes[phase]

                if phase == "train":
                    train_hist.append(epoch_loss)
                    logtxtbox.write(f"Train — Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")
                else:
                    val_hist.append(epoch_loss)
                    logtxtbox2.write(f"Val — Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_wts = copy.deepcopy(model.state_dict())

                tl.set_data(range(1, len(train_hist) + 1), train_hist)
                vl.set_data(range(1, len(val_hist) + 1), val_hist)
                ax.relim()
                ax.autoscale_view()
                with plot_spot:
                    st.pyplot(fig)

        elapsed = time.time() - since
        logtxtbox.write(
            f"Done in {elapsed // 60:.0f}m {elapsed % 60:.0f}s — Best val acc: {best_acc:.4f}"
        )
        model.load_state_dict(best_wts)
        return model

    def evaluate_classifier(model, dataloader, class_names):
        model.eval()
        all_labels, all_preds = [], []
        correct_pred = {c: 0 for c in class_names}
        total_pred   = {c: 0 for c in class_names}

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
                _, preds = torch.max(model(inputs), 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                for lbl, pred in zip(labels, preds):
                    if lbl == pred:
                        correct_pred[class_names[lbl]] += 1
                    total_pred[class_names[lbl]] += 1

        accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        f1    = f1_score(all_labels, all_preds, average="weighted")
        kappa = cohen_kappa_score(all_labels, all_preds)
        return accuracy, f1, kappa

    if start_validation and model_selection and dataset_path:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            transform_val = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(**IMAGENET_NORMALIZE),
            ])
            dataset     = datasets.ImageFolder(root=dataset_path, transform=transform_val)
            class_names = dataset.classes
            with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
                tmp.write(model_selection.read())
                tmp_path = tmp.name
            model = load_model(tmp_path, device=str(device))

            for param in model.encoder.parameters():
                param.requires_grad = False
            model.projection_head = nn.Sequential(
                nn.Dropout(),
                nn.Linear(model.feature_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, len(class_names)),
            )
            model = model.to(device)

            train_size = int(0.8 * len(dataset))
            val_size   = len(dataset) - train_size
            train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

            dataloaders  = {
                "train":      DataLoader(train_ds, batch_size=batch_size_validation, shuffle=True,  num_workers=4),
                "validation": DataLoader(val_ds,   batch_size=batch_size_validation, shuffle=False, num_workers=4),
            }
            dataset_sizes = {"train": train_size, "validation": val_size}

            st.write("Training linear probe…")
            model = train_linear_classifier(model, dataloaders, dataset_sizes)

            st.write("Evaluating…")
            acc, f1, kappa = evaluate_classifier(model, dataloaders["validation"], class_names)
            st.write(f"Accuracy: {acc:.2f}%")
            st.write(f"F1 score: {f1:.4f}")
            st.write(f"Cohen's κ: {kappa:.4f}")

        except Exception as e:
            st.error(f"Error: {e}")


# ===========================================================================
# GradCAM page
# ===========================================================================

elif page == "GradCAM":
    st.header("Grad-CAM Visualization")

    gradcam_mode  = st.radio("Input mode", ["Single Image", "Folder"], horizontal=True)
    gradcam_img   = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"]) if gradcam_mode == "Single Image" else None
    gradcam_folder = st.text_input("Image folder") if gradcam_mode == "Folder" else ""
    output_folder  = st.text_input("Output folder", "gradcam_results")
    model_selection   = st.file_uploader("Model file (.pth)", type=["pth"])
    run_gradcam    = st.button("Run Grad-CAM")

    if run_gradcam:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp.write(model_selection.read())
            tmp_path = tmp.name
        model = load_model(tmp_path, device=str(device)).to(device)
        model.eval()

        cam = GradCAM(model=model, target_layers=[model.encoder.layer3])
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(**IMAGENET_NORMALIZE),
        ])
        os.makedirs(output_folder, exist_ok=True)

        def process_image(image: Image.Image, filename: str):
            rgb_img      = np.array(image.resize((224, 224))).astype(np.float32) / 255
            input_tensor = transform(image).unsqueeze(0).to(device)
            grayscale    = cam(input_tensor=input_tensor, targets=None)[0]
            result       = Image.fromarray(show_cam_on_image(rgb_img, grayscale, use_rgb=True))
            result.save(os.path.join(output_folder, filename))
            return result

        if gradcam_mode == "Single Image" and gradcam_img is not None:
            result = process_image(Image.open(gradcam_img).convert("RGB"), "gradcam_result.png")
            st.image(result, caption="Grad-CAM Result")

        elif gradcam_mode == "Folder" and gradcam_folder:
            files = [
                os.path.join(root, f)
                for root, _, filenames in os.walk(gradcam_folder)
                for f in filenames
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            prog = st.progress(0.0, text="Running Grad-CAM…")
            for i, fpath in enumerate(files):
                stem = os.path.splitext(os.path.basename(fpath))[0]
                process_image(Image.open(fpath).convert("RGB"), f"{stem}_gradcam.png")
                prog.progress((i + 1) / len(files), text=f"{fpath}")
            st.success(f"Grad-CAM saved for {len(files)} images.")


# ===========================================================================
# Visualization page
# ===========================================================================

elif page == "Visualization":
    import umap
    from distinctipy import distinctipy
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    import plotly.graph_objects as go

    st.title("Embeddings Visualization & k-NN Evaluation")

    def hit_rate_at_k(X, y, k=5):
        nn_model = NearestNeighbors(n_neighbors=k + 1)
        nn_model.fit(X)
        indices = nn_model.kneighbors(X)[1][:, 1:]
        hits = np.array([int(y[i] in y[indices[i]]) for i in range(len(y))])
        overall = hits.mean()
        per_class = {lab: hits[y == lab].mean() for lab in np.unique(y)}
        return overall, per_class

    embeddings_csv = st.text_input("Embeddings CSV path", r"embeddings.csv")
    label_source   = st.radio("Label source", ["CSV file with labels", "Dataset folder structure"])

    labels_csv        = None
    dataset_folder    = None
    label_column_name = None

    if label_source == "CSV file with labels":
        labels_csv = st.text_input("Labels CSV path", r"labels.csv")
        if os.path.exists(labels_csv):
            tmp_df = pd.read_csv(labels_csv, sep=";")
            label_column_name = st.selectbox(
                "Label column",
                tmp_df.columns.tolist(),
                index=tmp_df.columns.tolist().index("label") if "label" in tmp_df.columns else 0,
            )
    else:
        dataset_folder = st.text_input("Dataset root folder (class subfolders)", value="")

    image_folder = st.text_input("Image folder (for overlay)", value="")
    method       = st.selectbox("Dimensionality reduction", ["pca", "tsne", "umap"])
    k            = st.number_input("k (k-NN classifier, CV)", value=5, min_value=1)
    k_nn         = st.number_input("k (hit-rate)", value=5, min_value=1)
    min_class_size = st.number_input("Min images per class (k-NN eval only)", value=1, min_value=1)

    run_button = st.button("Run Analysis")

    if run_button:
        emb = pd.read_csv(embeddings_csv, sep=";")
        emb["id"] = (
            emb.iloc[:, 0]
            .str.replace(".jpg", "", regex=False)
            .str.replace(".png", "", regex=False)
        )

        if label_source == "CSV file with labels":
            labels = pd.read_csv(labels_csv, sep=";")
            labels["id"] = labels["id"].astype(str)
            merged = emb.merge(labels, on="id", how="left")
            merged["label"] = merged[label_column_name]
        else:
            label_list = []
            for fname in emb.iloc[:, 0]:
                base = fname.replace(".jpg", "").replace(".png", "")
                detected = None
                for cls in os.listdir(dataset_folder):
                    p = os.path.join(dataset_folder, cls)
                    if not os.path.isdir(p):
                        continue
                    if (
                        os.path.exists(os.path.join(p, base + ".jpg"))
                        or os.path.exists(os.path.join(p, base + ".png"))
                    ):
                        detected = cls
                        break
                label_list.append(detected)
            emb["label"] = label_list
            merged = emb

        merged = merged.dropna(subset=["label"])

        X_all = np.nan_to_num(merged.iloc[:, 1:2049].values)
        y_all = merged["label"].values

        if method == "pca":
            reducer = PCA(n_components=2)
        elif method == "tsne":
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        else:
            reducer = umap.UMAP(n_components=2, random_state=42)

        proj = reducer.fit_transform(X_all)
        merged["x"], merged["y"] = proj[:, 0], proj[:, 1]

        unique_labels = merged["label"].unique()
        colors = distinctipy.get_colors(len(unique_labels))
        color_dict = {
            lab: f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"
            for lab, c in zip(unique_labels, colors)
        }
        merged["color"] = merged["label"].map(color_dict)

        fig = go.Figure()
        for lbl in unique_labels:
            s = merged[merged["label"] == lbl]
            fig.add_trace(go.Scatter(
                x=s["x"], y=s["y"],
                mode="markers",
                marker=dict(color=color_dict[lbl], size=20),
                name=lbl,
                text=s["id"],
                hovertemplate="<b>%{text}</b><br>Label: " + lbl + "<extra></extra>",
            ))

        for _, r in merged.iterrows():
            fp   = os.path.join(image_folder, r.iloc[0])
            fp2  = fp.replace(".jpg", ".png")
            final = fp2 if os.path.exists(fp2) else fp
            if os.path.exists(final):
                fig.add_layout_image(dict(
                    source=Image.open(final).convert("RGBA"),
                    x=r["x"], y=r["y"],
                    xref="x", yref="y",
                    xanchor="center", yanchor="middle",
                    sizex=1.5, sizey=1.5,
                    sizing="contain", layer="above", opacity=1,
                ))

        fig.update_layout(
            title=f"{method.upper()} Morphospace",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            hovermode="closest",
            width=1000, height=800,
        )
        st.plotly_chart(fig, use_container_width=True)

        df_eval = merged.copy()
        if min_class_size > 1:
            counts = df_eval["label"].value_counts()
            df_eval = df_eval[df_eval["label"].isin(counts[counts >= min_class_size].index)]

        if df_eval["label"].nunique() >= 2:
            X = df_eval.iloc[:, 1:2049].values
            y = df_eval["label"].values

            overall_hr, per_class_hr = hit_rate_at_k(X, y, k=k_nn)
            st.write(f"Hit-rate @ {k_nn} (overall): {overall_hr:.3f}")
            st.dataframe(
                pd.DataFrame.from_dict(per_class_hr, orient="index", columns=[f"Hit-rate@{k_nn}"])
            )

            knn_clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
            cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            y_pred  = cross_val_predict(knn_clf, X, y, cv=cv)

            st.write(f"k-NN accuracy (5-fold CV): {accuracy_score(y, y_pred):.3f}")
            per_label = {
                lab: accuracy_score(y[y == lab], y_pred[y == lab])
                for lab in np.unique(y)
            }
            st.dataframe(
                pd.DataFrame.from_dict(per_label, orient="index", columns=["kNN_accuracy"])
            )
        else:
            st.warning("Not enough classes with sufficient size for k-NN evaluation.")
