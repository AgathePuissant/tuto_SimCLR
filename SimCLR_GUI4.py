import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import glob
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
import time
import copy
from typing import Optional, Tuple, Union
import random
import torch.nn.functional as F
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import logging
from io import BytesIO
import base64
from datetime import datetime
from scipy.stats import mode

logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.caching").setLevel(logging.ERROR)

# === Backbone registry ===
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

# === Encoder Model ===
class Encoder(nn.Module):
    """
    ResNet encoder + projection head.
    Format de sauvegarde identique à train_supcon_hpc_v2 :
      ckpt = {"model_state", "backbone", "out_dim", "saved_at"}
    """
    def __init__(self, backbone: str = "resnet50", out_dim: int = 128, pretrained: bool = True):
        super().__init__()
        if backbone not in _BACKBONE_MAP:
            raise ValueError(f"Backbone '{backbone}' non supporté. Choix : {list(_BACKBONE_MAP.keys())}")
        self.backbone_name = backbone
        self.out_dim = out_dim
        self.feature_dim = _BACKBONE_FEAT_DIM[backbone]

        backbone_fn = _BACKBONE_MAP[backbone]
        # AFTER
        if pretrained:
    	    weights = "DEFAULT"
        else:
    	    weights = None
        self.encoder = backbone_fn(weights=weights)
        self.encoder.fc = nn.Identity()

        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.projection_head(features)

    def encode_backbone(self, x):
        """Renvoie les features 2048-d (sans projection head)."""
        return self.encoder(x)


def save_model(model: Encoder, path: str):
    """Sauvegarde au format enrichi (identique à train_supcon_hpc_v2)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "backbone":    model.backbone_name,
        "out_dim":     model.out_dim,
        "saved_at":    datetime.now().isoformat(),
    }
    torch.save(ckpt, path)


def save_full_checkpoint(model: Encoder, optimizer, epoch: int, save_path: str, hparams: dict):
    """Checkpoint complet avec état optimiseur et hyperparamètres."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    ckpt = {
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "epoch":           epoch,
        "hparams":         hparams,
        "saved_at":        datetime.now().isoformat(),
    }
    torch.save(ckpt, save_path)


def load_model(path: str, device="cpu") -> Encoder:
    """
    Charge un modèle depuis un fichier .pth.
    Supporte les deux formats :
      - dict enrichi : {"model_state", "backbone", "out_dim", ...}
      - state_dict brut (anciens modèles)
    """
    raw = torch.load(path, map_location=device)
    if isinstance(raw, dict) and "model_state" in raw:
        backbone = raw.get("backbone", "resnet50")
        out_dim  = raw.get("out_dim",  128)
        model = Encoder(backbone=backbone, out_dim=out_dim, pretrained=False)
        model.load_state_dict(raw["model_state"])
    else:
        # ancien format : state_dict brut
        model = Encoder(backbone="resnet50", out_dim=128, pretrained=False)
        model.load_state_dict(raw)
    return model


# === NT-Xent Loss — vectorisée multi-vues (identique à la logique du fichier de référence) ===
def nt_xent_loss_multi(z, batch_size, num_views, temperature=0.5):
    """
    z: (num_views * B, D) tensor — concatenated projections for all views
    batch_size: B
    num_views: N
    Implements: for each anchor, positives are other views of the same sample.
    Loss_i = -log( sum_{p in positives} exp(sim(i,p)/T) / sum_{j != i} exp(sim(i,j)/T) )
    Returns average loss over all anchors.
    """
    device = z.device
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.T) / temperature  # (N*B, N*B)
    # mask out self
    diag_mask = torch.eye(sim.size(0), dtype=torch.bool, device=device)
    sim_exp = torch.exp(sim.masked_fill(diag_mask, -9e15))  # exp(-inf)=0 for self

    # build sample ids: anchor index -> sample idx
    idxs = torch.arange(sim.size(0), device=device)
    sample_ids = idxs // num_views  # same sample id for all views of same original image

    # positive mask: same sample & not self
    positive_mask = (sample_ids.unsqueeze(0) == sample_ids.unsqueeze(1)) & (~diag_mask)

    # denominator: sum over all j != i
    denom = sim_exp.sum(dim=1)  # (N*B,)
    # numerator: sum of exp(sim) over positives
    pos_exp = (sim_exp * positive_mask.float()).sum(dim=1)  # (N*B,)

    # numerical stability: avoid zeros (shouldn't happen if num_views >= 2)
    eps = 1e-8
    loss = -torch.log((pos_exp + eps) / (denom + eps))
    return loss.mean()






# === SimCLR Transformations (multi-view) ===
class SimCLRTransform:
    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 0.5,
        min_scale: float = 0.08,
        #min_crop : float = 0.6,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[int] = None,
        sigmas: Tuple[float, float] = (0.2, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        normalize: dict = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
        num_views: int = 2
    ):
        if kernel_size is None:
            kernel_size = int(0.1 * input_size)
            if kernel_size % 2 == 0:
                kernel_size += 1

        transform_list = [
            transforms.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            #transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(min_crop, 1)),
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
            color_jitter = transforms.ColorJitter(
                brightness=cj_strength,
                contrast=cj_strength,
                saturation=cj_strength,
                hue=0.1 * cj_strength,
            )
            transform_list.append(transforms.RandomApply([color_jitter], p=cj_prob))

        if random_gray_scale > 0:
            transform_list.append(transforms.RandomGrayscale(p=random_gray_scale))

        if gaussian_blur > 0:
            transform_list.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigmas)],
                    p=gaussian_blur
                )
            )

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(**normalize)
        ])

        self.transform = transforms.Compose(transform_list)
        self.num_views = int(num_views)

    def __call__(self, x):
        # Return a list of N augmented views (each a tensor)
        return [self.transform(x) for _ in range(self.num_views)]


# === DataLoader collate function for multi-view batches ===
def simclr_collate(batch):
    """
    batch: list of tuples (views_list, label)
      views_list = [v1, v2, ..., vN] where each vi is a tensor [C,H,W]
    returns:
      views: list of N tensors each shaped [B, C, H, W]
      labels: tensor [B]
    """
    views_lists = [item[0] for item in batch]  # list of lists
    # transpose: list of tuples -> list of lists per view
    # views_per_view[i] is tuple of tensors of length B
    views_per_view = list(zip(*views_lists))
    views = [torch.stack(vs, dim=0) for vs in views_per_view]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return views, labels

# Update display_augmentations to handle list of views
def display_augmentations(dataset_path, transform, n=8):
    import random
    import matplotlib.pyplot as plt
    from PIL import Image

    imgs = []
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                imgs.append(os.path.join(root, f))
    if not imgs:
        st.warning("No images found in dataset folder.")
        return

    img_path = random.choice(imgs)
    img = Image.open(img_path).convert("RGB")

    # Show up to n augmentations (use multiple views if available)
    fig, axes = plt.subplots(1, n, figsize=(2*n, 2))
    for i in range(n):
        aug_views = transform(img)  # list of views
        # show first view (or rotate through)
        aug_tensor = aug_views[i % len(aug_views)]
        aug_img = aug_tensor.permute(1, 2, 0).numpy()
        aug_img = aug_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        aug_img = np.clip(aug_img, 0, 1)
        axes[i].imshow(aug_img)
        axes[i].axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    
@torch.no_grad()
def knn_cosine_accuracy(train_feats, train_labels, val_feats, val_labels, k=1, device='cuda'):
    """
    Efficient k-NN accuracy computation using cosine similarity on GPU.
    - train_feats: (N_train, D) tensor
    - train_labels: (N_train,) tensor
    - val_feats: (N_val, D) tensor
    - val_labels: (N_val,) tensor
    """

    # Move everything to GPU if available
    train_feats = train_feats.to(device)
    val_feats = val_feats.to(device)
    train_labels = train_labels.to(device)
    val_labels = val_labels.to(device)

    # Normalize for cosine similarity
    train_norm = torch.nn.functional.normalize(train_feats, dim=1)
    val_norm = torch.nn.functional.normalize(val_feats, dim=1)

    # Cosine similarity = dot product of normalized vectors
    sims = torch.mm(val_norm, train_norm.T)   # (n_val, n_train)

    # Top-k neighbors
    topk = sims.topk(k, dim=1)
    topk_idx = topk.indices  # (n_val, k)
    topk_labels = train_labels[topk_idx]  # gather labels

    # Majority vote (vectorized)
    if k == 1:
        preds = topk_labels.squeeze(1)
    else:
        # For k>1, find the most frequent label per row
        preds = torch.mode(topk_labels, dim=1).values

    # Compute accuracy
    acc = (preds == val_labels).float().mean().item()
    return acc


# === Streamlit UI ===
st.title("SimCLR Training GUI")

page = st.sidebar.radio(
    "Select a page:",
    ["Training", "SupCon Training", "Generate Embeddings", "Validation", "GradCAM", "Visualization"]
)


# === Training Page ===
if page == "Training":
    st.header("SimCLR Training")
    
    # User Inputs
    dataset_path = st.text_input("Enter Dataset Path", value="data")
    save_model_path = st.text_input("Enter Model Save Path (file)", value="runs/simclr_model.pth")
    batch_size = st.slider("Batch Size", 16, 256, 128)
    epochs = st.slider("Epochs", 1, 300, 300)
    learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-1, value=1e-3, format="%.6f")

    # Checkpoint & best-model options
    checkpoint_freq = st.number_input("Checkpoint every N epochs", min_value=1, max_value=100, value=10, step=1)
    enable_knn = st.checkbox("Enable k-NN evaluation at checkpoints (requires labeled dataset)", value=False)
    labeled_dataset_path = st.text_input("Labeled dataset path (subfolders = classes) - for k-NN", value="")
    knn_k = st.slider("k for k-NN (majority vote)", 1, 15, 5)
    save_best_model = st.checkbox("Save best model (based on k-NN acc if enabled, else based on lowest loss)", value=False)

    # Image Transformation Probabilities
    input_size = st.number_input("Input Size", min_value=64, max_value=512, value=224, step=8)
    min_scale = st.slider("Minimum Scale for RandomResizedCrop", 0.01, 1.0, 0.3, 0.01)
    cj_col = st.columns(2)
    with cj_col[0]:
        cj_prob = st.slider("Color Jitter Probability", 0.0, 1.0, 0.5, 0.01)
    with cj_col[1]:
        cj_strength = st.slider("Color Jitter Strength", 0.0, 1.0, 0.3, 0.01)
    flip_col = st.columns(2)
    with flip_col[0]:
        hf_prob = st.slider("Horizontal Flip Probability", 0.0, 1.0, 0.5, 0.01)
    with flip_col[1]:
        vf_prob = st.slider("Vertical Flip Probability", 0.0, 1.0, 0.5, 0.01)
    gray_blur_col = st.columns(2)
    with gray_blur_col[0]:
        random_grayscale_prob = st.slider("Random Grayscale Probability", 0.0, 1.0, 0.3, 0.01)
    with gray_blur_col[1]:
        gaussian_blur_prob = st.slider("Gaussian Blur Probability", 0.0, 1.0, 0.3, 0.01)
    blur_col = st.columns(2)
    with blur_col[0]:
        sigmas_min = st.number_input("Min Sigma for Gaussian Blur", 0.01, 5.0, 0.2, 0.01)
    with blur_col[1]:
        sigmas_max = st.number_input("Max Sigma for Gaussian Blur", 0.1, 5.0, 2.0, 0.01)
    rotation = st.checkbox("Enable Random Rotation", value=False)
    rr_prob = st.slider("Random Rotation Probability", 0.0, 1.0, 0.5, 0.01) if rotation else 0.0
    rr_degrees = st.slider("Rotation Degrees Range", 0, 180, 45, 1) if rotation else None

    normalize = {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
    
    num_views = st.slider("Number of views per image (N)", 2, 6, 2)  # min 2, max 6, default 2
    temperature = st.number_input("NT-Xent temperature", min_value=0.01, max_value=1.0, value=0.8, step=0.01, format="%.2f")
    backbone_simclr = st.selectbox("Backbone", list(_BACKBONE_MAP.keys()), index=2)  # resnet50 par défaut
    out_dim_simclr  = st.number_input("Projection head output dim", min_value=32, max_value=512, value=128, step=32)

    show_aug = st.button("Show Random Augmentations")
    start_training = st.button("Start Training")


    # Update augment_params_dict to include num_views and temperature
    def augment_params_dict():
        return {
            "input_size": input_size,
            "min_scale": min_scale,
            "cj_prob": cj_prob,
            "cj_strength": cj_strength,
            "hf_prob": hf_prob,
            "vf_prob": vf_prob,
            "random_grayscale_prob": random_grayscale_prob,
            "gaussian_blur_prob": gaussian_blur_prob,
            "sigmas_min": sigmas_min,
            "sigmas_max": sigmas_max,
            "rr_prob": rr_prob,
            "rr_degrees": rr_degrees,
            "num_views": num_views,
            "temperature": temperature
        }
    
    
    # === Training function changes (core training loop) ===
    def train_simclr_gui(model, dataloader, epochs, lr, device, transform, save_model_path,
                         checkpoint_freq=10, enable_knn=False, labeled_dataset_path=None,
                         knn_k=1, save_best_model=True, augment_params=None, dataset_name=None,
                         temperature=0.5, num_views=2):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.to(device)
        loss_history = []
    
        plot_spot = st.empty()
        logtxtbox = st.empty()
    
        fig, ax = plt.subplots()
        loss_line, = ax.plot([], [], marker="o", label="Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Progress")
        ax.legend()
    
        # prepare folders and log path
        model_dir = os.path.dirname(save_model_path) if os.path.dirname(save_model_path) else "."
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_dir = os.path.join(model_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_csv_path = os.path.join(model_dir, os.path.splitext(os.path.basename(save_model_path))[0] + "_training_log.csv")
    
        # initialize log df (will overwrite if exists)
        log_columns = ["epoch", "loss", "knn_acc", "lr", "batch_size", "dataset_name", "augment_params", "timestamp"]
        log_df = pd.DataFrame(columns=log_columns)
        log_df.to_csv(log_csv_path, index=False, sep=";")
    
        best_metric = -1.0  # will track best knn accuracy or negative loss if knn disabled
        best_loss = float("inf")
        best_epoch = -1
    
        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
            for views, _ in progress_bar:
                # # views: list of num_views tensors each [B, C, H, W]
                # # move to device
                # views = [v.to(device) for v in views]
                # # compute projections for each view
                # zs = [model(v) for v in views]  # list of [B, D]
                # # concatenate along batch dim -> (N*B, D)
                # z = torch.cat(zs, dim=0)
                # loss = nt_xent_loss_multi(z, batch_size=zs[0].size(0), num_views=len(zs), temperature=temperature)
                
                # model outputs embeddings for all views in a list
                view_embs = [model(view.to(device)) for view in views]  # views = list of tensors
                z = torch.cat(view_embs, dim=0)   # (num_views * B, D)
                loss = nt_xent_loss_multi(
                    z,
                    batch_size=view_embs[0].size(0),
                    num_views=len(view_embs),
                    temperature=temperature,
                )

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
    
            logtxtbox.write(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
            with plot_spot:
                st.pyplot(fig)
    
            # Default knn_acc None
            knn_acc = None
    
            # Checkpointing + optional k-NN eval
            if (epoch + 1) % checkpoint_freq == 0 or (epoch + 1) == epochs:
                # save checkpoint
                chk_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
                save_full_checkpoint(model, optimizer, epoch + 1, chk_path, hparams=augment_params or {})
                st.write(f"Saved checkpoint: {chk_path}")
    
                # if enabled and labeled_dataset_path provided, compute embeddings and knn accuracy
                if enable_knn and labeled_dataset_path and os.path.isdir(labeled_dataset_path):
                    st.write("Running k-NN evaluation on labeled dataset...")
                    try:
                        # safe resize for eval transform
                        if hasattr(transform.transform.transforms[0], 'size'):
                            sz = transform.transform.transforms[0].size
                            if isinstance(sz, int):
                                sz = (sz, sz)
                            elif isinstance(sz, tuple):
                                if len(sz) != 2:
                                    raise ValueError(f"Expected tuple of length 2 for size, got {sz}")
                        else:
                            sz = (input_size, input_size)
                        
                        transform_eval = transforms.Compose([
                            transforms.Resize(sz),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=normalize['mean'], std=normalize['std'])
                        ])
    
                        labeled_dataset = datasets.ImageFolder(root=labeled_dataset_path, transform=transform_eval)
                        if len(labeled_dataset) < 2:
                            st.warning("Labeled dataset needs at least 2 images.")
                        else:
                            # create train/val split indices
                            n = len(labeled_dataset)
                            indices = list(range(n))
                            random.shuffle(indices)
                            split = int(0.8 * n)
                            train_idx, val_idx = indices[:split], indices[split:]
                            # Create subset loaders
                            from torch.utils.data import Subset
                            train_sub = Subset(labeled_dataset, train_idx)
                            val_sub = Subset(labeled_dataset, val_idx)
                            train_loader = DataLoader(train_sub, batch_size=64, shuffle=False, num_workers=4)
                            val_loader = DataLoader(val_sub, batch_size=64, shuffle=False, num_workers=4)
    
                            # extract embeddings
                            model.eval()
                            train_feats = []
                            train_labels = []
                            with torch.no_grad():
                                for imgs, labels in train_loader:
                                    imgs = imgs.to(device)
                                    feats = model.encode_backbone(imgs).cpu().numpy()
                                    train_feats.append(feats)
                                    train_labels.extend(labels.numpy().tolist())
                            val_feats = []
                            val_labels = []
                            with torch.no_grad():
                                for imgs, labels in val_loader:
                                    imgs = imgs.to(device)
                                    feats = model.encode_backbone(imgs).cpu().numpy()
                                    val_feats.append(feats)
                                    val_labels.extend(labels.numpy().tolist())
                            if len(train_feats) == 0 or len(val_feats) == 0:
                                knn_acc = None
                                st.warning("k-NN evaluation skipped due to empty train/val splits.")
                            else:
                                train_feats = np.vstack(train_feats)
                                val_feats = np.vstack(val_feats)
                                
                                # Convert once to tensors before calling
                                train_feats_t = torch.tensor(train_feats, dtype=torch.float32)
                                val_feats_t = torch.tensor(val_feats, dtype=torch.float32)
                                train_labels_t = torch.tensor(train_labels, dtype=torch.long)
                                val_labels_t = torch.tensor(val_labels, dtype=torch.long)
                                
                                knn_acc = knn_cosine_accuracy(train_feats_t, train_labels_t, val_feats_t, val_labels_t, k=knn_k, device=device)
                                
                                if knn_acc is not None:
                                    st.write(f"k-NN accuracy (k={knn_k}): {knn_acc*100:.2f}%")
                                else:
                                    st.write("k-NN returned no result.")
                    except Exception as e:
                        st.write(f"k-NN evaluation failed: {e}")
                        knn_acc = None
    
                # Update best model logic
                if enable_knn and knn_acc is not None:
                    metric_val = knn_acc
                    if metric_val > best_metric:
                        best_metric = metric_val
                        best_epoch = epoch + 1
                        if save_best_model:
                            best_path = os.path.join(model_dir, "best_model.pth")
                            save_model(model, best_path)
                            st.write(f"New best model saved (k-NN acc={best_metric:.4f}) -> {best_path}")
                else:
                    # Use loss (lower is better)
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_epoch = epoch + 1
                        if save_best_model:
                            best_path = os.path.join(model_dir, "best_model.pth")
                            save_model(model, best_path)
                            st.write(f"New best model saved (loss={best_loss:.6f}) -> {best_path}")
    
            # Write log row
            log_row_dict = {
                "epoch": epoch+1,
                "loss": avg_loss,
                "knn_acc": knn_acc if knn_acc is not None else "",
                "lr": lr,
                "batch_size": batch_size,
                "dataset_name": dataset_name if dataset_name else os.path.basename(save_model_path),
                "augment_params": str(augment_params) if augment_params else "",
                "timestamp": datetime.now().isoformat()
            }
            log_row = pd.DataFrame([log_row_dict])  # convert dict to single-row DataFrame
            try:
                prev = pd.read_csv(log_csv_path, sep=";")
            except Exception:
                prev = pd.DataFrame(columns=log_columns)
            prev = pd.concat([prev, log_row], ignore_index=True)
            prev.to_csv(log_csv_path, index=False, sep=";")
    
        # final save (last model)
        final_path = os.path.join(model_dir, os.path.splitext(os.path.basename(save_model_path))[0] + "_last.pth")
        save_model(model, final_path)
        st.success(f"Final model saved to {final_path}")
        st.info(f"Best epoch: {best_epoch} (best_metric={best_metric if enable_knn else best_loss})")
        return model
    
    
    # === Start Training When Button Clicked ===
    if show_aug and dataset_path:
        try:
            transform = SimCLRTransform(
                input_size=input_size,
                cj_prob=cj_prob,
                cj_strength=cj_strength,
                min_scale=min_scale,
                random_gray_scale=random_grayscale_prob,
                gaussian_blur=gaussian_blur_prob,
                sigmas=(sigmas_min, sigmas_max),
                vf_prob=vf_prob,
                hf_prob=hf_prob,
                rr_prob=rr_prob,
                rr_degrees=(-rr_degrees, rr_degrees) if rr_degrees else None,
                normalize=normalize,
                num_views=num_views
            )
            display_augmentations(dataset_path, transform, n=8)
        except Exception as e:
            st.error(f"Could not display augmentations: {e}")
    
    if start_training and dataset_path:
        try:
            st.write("Loading dataset...")
            transform = SimCLRTransform(
                input_size=input_size,
                cj_prob=cj_prob,
                cj_strength=cj_strength,
                min_scale=min_scale,
                random_gray_scale=random_grayscale_prob,
                gaussian_blur=gaussian_blur_prob,
                sigmas=(sigmas_min, sigmas_max),
                vf_prob=vf_prob,
                hf_prob=hf_prob,
                rr_prob=rr_prob,
                rr_degrees=(-rr_degrees, rr_degrees) if rr_degrees else None,
                normalize=normalize,
                num_views=num_views
            )
    
            # Use ImageFolder for unlabeled dataset (transform returns list of views per image)
            dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    
            # Count class occurrences and weighted sampler (unchanged)
            class_counts = [0] * len(dataset.classes)
            for _, label in dataset.samples:
                class_counts[label] += 1
            class_weights = [1.0 / count if count > 0 else 0.0 for count in class_counts]
            weights = [class_weights[label] for _, label in dataset.samples]
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
            dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, collate_fn=simclr_collate)
            st.write(f"Dataset Loaded! {len(dataset)} images found.")
            model = Encoder(backbone=backbone_simclr, out_dim=int(out_dim_simclr))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            train_simclr_gui(
                model=model,
                dataloader=dataloader,
                epochs=epochs,
                lr=learning_rate,
                device=device,
                transform=transform,
                save_model_path=save_model_path,
                checkpoint_freq=int(checkpoint_freq),
                enable_knn=enable_knn,
                labeled_dataset_path=labeled_dataset_path,
                knn_k=int(knn_k),
                save_best_model=save_best_model,
                augment_params=augment_params_dict(),
                dataset_name=os.path.basename(dataset_path.rstrip("/")),
                temperature=temperature,
                num_views=num_views
            )
        except Exception as e:
            st.error(f"Error: {str(e)}")


# ===========================
# ===== SupCon Page =========
# ===========================
if page == "SupCon Training":
    st.header("Supervised Contrastive (SupCon) Training")

    # --- Inputs (mirror SimCLR page UI) ---
    dataset_path_sup = st.text_input("Dataset path (subfolders = classes)", value="")
    save_model_path_sup = st.text_input("Save model as:", value="supcon_model.pth")
    batch_size_sup = st.slider("Batch Size", 8, 256, 32)
    epochs_sup = st.slider("Epochs", 1, 300, 20)
    learning_rate_sup = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-1, value=1e-3, format="%.6f")

    checkpoint_freq_sup = st.number_input("Checkpoint every N epochs", min_value=1, max_value=100, value=10, step=1)
    enable_knn_sup = st.checkbox("Enable k-NN evaluation at checkpoints (requires labeled dataset)", value=False)
    knn_k_sup = st.slider("k for k-NN (majority vote)", 1, 15, 1)
    save_best_model_sup = st.checkbox("Save best model (based on k-NN acc if enabled, else based on lowest loss)", value=True)

    backbone_sup = st.selectbox("Backbone", list(_BACKBONE_MAP.keys()), index=2)  # resnet50 par défaut
    out_dim_sup  = st.number_input("Projection head output dim", min_value=32, max_value=512, value=128, step=32)

    # Image Transformation Probabilities
    input_size = st.number_input("Input Size", min_value=64, max_value=512, value=224, step=8)
    min_scale = st.slider("Minimum Scale for RandomResizedCrop", 0.01, 1.0, 0.3, 0.01)
    cj_col = st.columns(2)
    with cj_col[0]:
        cj_prob = st.slider("Color Jitter Probability", 0.0, 1.0, 0.5, 0.01)
    with cj_col[1]:
        cj_strength = st.slider("Color Jitter Strength", 0.0, 1.0, 0.3, 0.01)
    flip_col = st.columns(2)
    with flip_col[0]:
        hf_prob = st.slider("Horizontal Flip Probability", 0.0, 1.0, 0.5, 0.01)
    with flip_col[1]:
        vf_prob = st.slider("Vertical Flip Probability", 0.0, 1.0, 0.5, 0.01)
    gray_blur_col = st.columns(2)
    with gray_blur_col[0]:
        random_grayscale_prob = st.slider("Random Grayscale Probability", 0.0, 1.0, 0.3, 0.01)
    with gray_blur_col[1]:
        gaussian_blur_prob = st.slider("Gaussian Blur Probability", 0.0, 1.0, 0.3, 0.01)
    blur_col = st.columns(2)
    with blur_col[0]:
        sigmas_min = st.number_input("Min Sigma for Gaussian Blur", 0.01, 5.0, 0.2, 0.01)
    with blur_col[1]:
        sigmas_max = st.number_input("Max Sigma for Gaussian Blur", 0.1, 5.0, 2.0, 0.01)
    rotation = st.checkbox("Enable Random Rotation", value=False)
    rr_prob = st.slider("Random Rotation Probability", 0.0, 1.0, 0.5, 0.01) if rotation else 0.0
    rr_degrees = st.slider("Rotation Degrees Range", 0, 180, 45, 1) if rotation else None


    normalize_sup = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    # SupCon-specific
    num_views_sup = st.slider("Number of views per image (N)", 2, 6, 2)
    temperature_sup = st.number_input("SupCon temperature (τ)", min_value=0.01, max_value=1.0, value=0.8, step=0.01, format="%.2f")

    show_aug_sup = st.button("Show Random Augmentations (SupCon)")
    start_supcon = st.button("Start SupCon Training")

    # helper: params dict (for logging)
    def augment_params_sup_dict():
        return {
            "input_size": input_size_sup,
            "min_scale": min_scale_sup,
            #"min_crop" : min_crop_sup,
            "cj_prob": cj_prob_sup,
            "cj_strength": cj_strength_sup,
            "hf_prob": hf_prob_sup,
            "vf_prob": vf_prob_sup,
            "random_grayscale_prob": random_grayscale_prob_sup,
            "gaussian_blur_prob": gaussian_blur_prob_sup,
            "sigmas_min": sigmas_min_sup,
            "sigmas_max": sigmas_max_sup,
            "rr_prob": rr_prob_sup,
            "rr_degrees": rr_degrees_sup,
            "num_views": num_views_sup,
            "temperature": temperature_sup
        }

    # display augmentations (reuse SimCLRTransform)
    if show_aug_sup and dataset_path_sup:
        try:
            tr_show = SimCLRTransform(
                input_size=input_size,
                cj_prob=cj_prob,
                cj_strength=cj_strength,
                min_scale=min_scale,
                #min_crop=min_crop,
                random_gray_scale=random_grayscale_prob,
                gaussian_blur=gaussian_blur_prob,
                sigmas=(sigmas_min, sigmas_max),
                vf_prob=vf_prob,
                hf_prob=hf_prob,
                rr_prob=rr_prob,
                rr_degrees=(-rr_degrees, rr_degrees) if rr_degrees else None,
                normalize=normalize_sup,
                num_views=num_views_sup
            )
            display_augmentations(dataset_path_sup, tr_show, n=8)
        except Exception as e:
            st.error(f"Could not display augmentations: {e}")

    # === SupCon loss implementation (memory-friendly, supports multi-view) ===
    class SupConLossLite(nn.Module):
        """
        SupCon loss that computes supervised contrastive objective.
        This version avoids building huge intermediate masks where possible
        and will fall back to sampled positive pairs if memory becomes an issue.
        """
        def __init__(self, temperature=0.07, use_all_pairs=True, samples_per_image=2):
            super().__init__()
            self.temperature = temperature
            self.use_all_pairs = use_all_pairs
            self.samples_per_image = samples_per_image

        def forward(self, features, labels):
            """
            features: list of V tensors each [B, D] (projection head outputs)
            labels: [B] long tensor
            """
            device = features[0].device
            labels = labels.to(device)
            B = labels.shape[0]
            V = len(features)

            # concatenate -> (V*B, D)
            feats = torch.cat(features, dim=0)
            feats = F.normalize(feats, dim=1)

            if self.use_all_pairs:
                # Build mask for positives across views (B x B) expanded to (VB x VB)
                mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)  # BxB
                pos_mask = mask.repeat(V, V)  # (VB x VB)
                self_mask = torch.eye(V * B, device=device).bool()
                pos_mask = pos_mask.masked_fill(self_mask, 0)

                sim = torch.matmul(feats, feats.T) / self.temperature
                exp_sim = torch.exp(sim)
                # zero diagonal in denominator
                exp_sim = exp_sim.masked_fill(self_mask, 0.0)
                denom = exp_sim.sum(dim=1)
                pos_sim = (exp_sim * pos_mask).sum(dim=1)
                loss = -torch.log((pos_sim + 1e-12) / (denom + 1e-12))
                return loss.mean()
            else:
                # Memory-friendly sampled pairs (use samples_per_image)
                # Memory-friendly sampled pairs (same-class positives)
                # Build per-view stack [V, B, D]
                views = torch.stack(features, dim=0)  # V x B x D
                views = F.normalize(views, dim=-1)
                all_embs = views.reshape(-1, views.shape[-1])  # (V*B, D)

                total_loss = 0.0
                total_pairs = 0
                for b in range(B):
                    # for each sample, sample pairs where the second view may come from any image of same class
                    same_idx = (labels == labels[b]).nonzero(as_tuple=False).squeeze(1)  # indices in [0..B-1]
                    # ensure we have >0 positives (at least itself)
                    if same_idx.numel() <= 1:
                        # fallback to random within-batch (exclude itself)
                        same_idx = torch.tensor([i for i in range(B) if i != b], device=device)
                        if same_idx.numel() == 0:
                            continue
                    for _ in range(self.samples_per_image):
                        # pick view i for anchor b
                        i = torch.randint(0, V, (1,)).item()
                        # pick positive sample index p_idx != b (if possible) otherwise allow b with different view
                        p_sample = same_idx[torch.randint(0, same_idx.numel(), (1,)).item()].item()
                        j = torch.randint(0, V, (1,)).item()
                        zi = views[i, b]   # [D]
                        zj = views[j, p_sample]
                        pos_sim = torch.matmul(zi, zj) / self.temperature
                        logits = torch.matmul(zi, all_embs.T) / self.temperature
                        pos_index = i * B + b
                        # mask out positive location (we'll keep negatives except that location)
                        logits = torch.cat([logits[:pos_index], logits[pos_index+1:]])
                        loss = -pos_sim + torch.logsumexp(logits, dim=0)
                        total_loss += loss
                        total_pairs += 1
                if total_pairs == 0:
                    return torch.tensor(0.0, device=device, requires_grad=True)
                return total_loss / total_pairs

    # === Training function for SupCon (mirrors SimCLR train function) ===
    def train_supcon_gui(model, dataloader, epochs, lr, device, transform, save_model_path,
                         checkpoint_freq=10, enable_knn=False, knn_k=1, save_best_model=True,
                         augment_params=None, dataset_name=None, temperature=0.07, num_views=2,
                         samples_per_image=2):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.to(device)
        loss_history = []

        plot_spot = st.empty()
        logtxtbox = st.empty()

        fig, ax = plt.subplots()
        loss_line, = ax.plot([], [], marker="o", label="Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("SupCon Loss Progress")
        ax.legend()

        # prepare folders and log path
        model_dir = os.path.dirname(save_model_path) if os.path.dirname(save_model_path) else "."
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_dir = os.path.join(model_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_csv_path = os.path.join(model_dir, os.path.splitext(os.path.basename(save_model_path))[0] + "_supcon_training_log.csv")

        # initialize log df (will overwrite if exists)
        log_columns = ["epoch", "loss", "knn_acc", "lr", "batch_size", "dataset_name", "augment_params", "timestamp"]
        log_df = pd.DataFrame(columns=log_columns)
        log_df.to_csv(log_csv_path, index=False, sep=";")

        best_metric = -1.0
        best_loss = float("inf")
        best_epoch = -1

        # choose whether to use all-pairs or sampled fallback based on memory heuristics
        use_all_pairs = True
        if num_views * dataloader.batch_size > 512:  # heuristic; adjust if needed
            use_all_pairs = False

        criterion = SupConLossLite(temperature=temperature, use_all_pairs=False, samples_per_image=samples_per_image)

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
            for views, labels in progress_bar:
                views = [v.to(device) for v in views]
                labels = labels.to(device)

                # compute projections for each view
                feats = [model(v) for v in views]  # [B, D] each

                loss = criterion(feats, labels)

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

            logtxtbox.write(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            with plot_spot:
                st.pyplot(fig)

            knn_acc = None

            # checkpoints and optional k-NN evaluation
            if (epoch + 1) % checkpoint_freq == 0 or (epoch + 1) == epochs:
                chk_path = os.path.join(checkpoint_dir, f"supcon_checkpoint_epoch_{epoch+1}.pth")
                save_full_checkpoint(model, optimizer, epoch + 1, chk_path, hparams=augment_params or {})
                st.write(f"Saved checkpoint: {chk_path}")

                if enable_knn and os.path.isdir(dataset_path_sup):
                    st.write("Running k-NN evaluation on labeled dataset...")
                    try:
                        # eval transform same as before
                        if hasattr(transform.transform.transforms[0], 'size'):
                            sz = transform.transform.transforms[0].size
                            if isinstance(sz, int):
                                sz = (sz, sz)
                            elif isinstance(sz, tuple):
                                if len(sz) != 2:
                                    raise ValueError(f"Expected tuple of length 2 for size, got {sz}")
                        else:
                            sz = (input_size_sup, input_size_sup)

                        transform_eval = transforms.Compose([
                            transforms.Resize(sz),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=normalize_sup['mean'] if isinstance(normalize_sup, dict) else normalize_sup['mean'],
                                                 std=normalize_sup['std'] if isinstance(normalize_sup, dict) else normalize_sup['std'])
                        ])

                        labeled_dataset = datasets.ImageFolder(root=dataset_path_sup, transform=transform_eval)
                        if len(labeled_dataset) < 2:
                            st.warning("Labeled dataset needs at least 2 images.")
                        else:
                            n = len(labeled_dataset)
                            indices = list(range(n))
                            random.shuffle(indices)
                            split = int(0.8 * n)
                            train_idx, val_idx = indices[:split], indices[split:]
                            from torch.utils.data import Subset
                            train_sub = Subset(labeled_dataset, train_idx)
                            val_sub = Subset(labeled_dataset, val_idx)
                            train_loader = DataLoader(train_sub, batch_size=64, shuffle=False, num_workers=4)
                            val_loader = DataLoader(val_sub, batch_size=64, shuffle=False, num_workers=4)

                            model.eval()
                            train_feats = []
                            train_labels = []
                            with torch.no_grad():
                                for imgs, labels_eval in train_loader:
                                    imgs = imgs.to(device)
                                    feats = model.encode_backbone(imgs).cpu().numpy()
                                    train_feats.append(feats)
                                    train_labels.extend(labels_eval.numpy().tolist())
                            val_feats = []
                            val_labels = []
                            with torch.no_grad():
                                for imgs, labels_eval in val_loader:
                                    imgs = imgs.to(device)
                                    feats = model.encode_backbone(imgs).cpu().numpy()
                                    val_feats.append(feats)
                                    val_labels.extend(labels_eval.numpy().tolist())
                            if len(train_feats) == 0 or len(val_feats) == 0:
                                knn_acc = None
                                st.warning("k-NN evaluation skipped due to empty train/val splits.")
                            else:
                                train_feats = np.vstack(train_feats)
                                val_feats = np.vstack(val_feats)
                                train_feats_t = torch.tensor(train_feats, dtype=torch.float32)
                                val_feats_t = torch.tensor(val_feats, dtype=torch.float32)
                                train_labels_t = torch.tensor(train_labels, dtype=torch.long)
                                val_labels_t = torch.tensor(val_labels, dtype=torch.long)

                                knn_acc = knn_cosine_accuracy(train_feats_t, train_labels_t, val_feats_t, val_labels_t, k=knn_k_sup, device=device)
                                if knn_acc is not None:
                                    st.write(f"k-NN accuracy (k={knn_k_sup}): {knn_acc*100:.2f}%")
                    except Exception as e:
                        st.write(f"k-NN evaluation failed: {e}")
                        knn_acc = None

                # best model logic
                if enable_knn and knn_acc is not None:
                    metric_val = knn_acc
                    if metric_val > best_metric:
                        best_metric = metric_val
                        best_epoch = epoch + 1
                        if save_best_model_sup:
                            best_path = os.path.join(model_dir, "supcon_best_model.pth")
                            save_model(model, best_path)
                            st.write(f"New best model saved (k-NN acc={best_metric:.4f}) -> {best_path}")
                else:
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_epoch = epoch + 1
                        if save_best_model_sup:
                            best_path = os.path.join(model_dir, "supcon_best_model.pth")
                            save_model(model, best_path)
                            st.write(f"New best model saved (loss={best_loss:.6f}) -> {best_path}")

            # logging row
            log_row_dict = {
                "epoch": epoch+1,
                "loss": avg_loss,
                "knn_acc": knn_acc if knn_acc is not None else "",
                "lr": lr,
                "batch_size": batch_size_sup,
                "dataset_name": dataset_name if dataset_name else os.path.basename(save_model_path_sup),
                "augment_params": str(augment_params) if augment_params else "",
                "timestamp": datetime.now().isoformat()
            }
            log_row = pd.DataFrame([log_row_dict])
            try:
                prev = pd.read_csv(log_csv_path, sep=";")
            except Exception:
                prev = pd.DataFrame(columns=log_columns)
            prev = pd.concat([prev, log_row], ignore_index=True)
            prev.to_csv(log_csv_path, index=False, sep=";")

        # final save
        final_path = os.path.join(model_dir, os.path.splitext(os.path.basename(save_model_path_sup))[0] + "_last.pth")
        save_model(model, final_path)
        st.success(f"Final model saved to {final_path}")
        st.info(f"Best epoch: {best_epoch} (best_metric={best_metric if enable_knn else best_loss})")
        return model

    # === Start SupCon training when button clicked ===
    if start_supcon and dataset_path_sup:
        try:
            st.write("Loading dataset for SupCon...")
            transform_sup = SimCLRTransform(
                input_size=input_size,
                cj_prob=cj_prob,
                cj_strength=cj_strength,
                min_scale=min_scale,
                random_gray_scale=random_grayscale_prob,
                gaussian_blur=gaussian_blur_prob,
                sigmas=(sigmas_min, sigmas_max),
                vf_prob=vf_prob,
                hf_prob=hf_prob,
                rr_prob=rr_prob,
                rr_degrees=(-rr_degrees, rr_degrees) if rr_degrees else None,
                normalize=normalize_sup,
                num_views=num_views_sup
            )

            # ImageFolder dataset (folder labels)
            dataset_sup = datasets.ImageFolder(root=dataset_path_sup, transform=transform_sup)
            if len(dataset_sup) == 0:
                st.error("No images found in dataset path.")
            else:
                # compute class counts for weighted sampler
                class_counts_sup = [0] * len(dataset_sup.classes)
                for _, lab in dataset_sup.samples:
                    class_counts_sup[lab] += 1
                class_weights_sup = [1.0 / c if c > 0 else 0.0 for c in class_counts_sup]
                weights_sup = [class_weights_sup[lab] for _, lab in dataset_sup.samples]
                sampler_sup = WeightedRandomSampler(weights_sup, len(weights_sup), replacement=True)

                dataloader_sup = DataLoader(dataset_sup, batch_size=batch_size_sup, sampler=sampler_sup,
                                            num_workers=4, collate_fn=simclr_collate)

                st.write(f"SupCon Dataset Loaded! {len(dataset_sup)} images, {len(dataset_sup.classes)} classes.")
                model_sup = Encoder(backbone=backbone_sup, out_dim=out_dim_sup)
                device_sup = "cuda" if torch.cuda.is_available() else "cpu"

                # Train
                train_supcon_gui(
                    model=model_sup,
                    dataloader=dataloader_sup,
                    epochs=epochs_sup,
                    lr=learning_rate_sup,
                    device=device_sup,
                    transform=transform_sup,
                    save_model_path=save_model_path_sup,
                    checkpoint_freq=int(checkpoint_freq_sup),
                    enable_knn=enable_knn_sup,
                    knn_k=knn_k_sup,
                    save_best_model=save_best_model_sup,
                    augment_params=augment_params_sup_dict(),
                    dataset_name=os.path.basename(dataset_path_sup.rstrip("/")),
                    temperature=temperature_sup,
                    num_views=num_views_sup,
                    samples_per_image=samples_per_image_sup
                )
        except Exception as e:
            st.error(f"SupCon training failed: {str(e)}")
            
# === Embeddings Page ===
elif page == "Generate Embeddings":
    st.header("Generate Embeddings")
    embeddings_folder = st.text_input("Enter Folder Path for Embeddings", value="")
    model_selection = st.file_uploader("Upload Model for Embeddings (.pth)", type=["pth"])
    embedding_save_path = st.text_input("Enter Path to Save Embeddings CSV", value="embeddings.csv")
    image_paths = []
    for root, dirs, filenames in os.walk(embeddings_folder):
        for f in filenames:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(root, f))
    start_embedding = st.button("Generate Embeddings")

    if start_embedding and embeddings_folder and model_selection:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Sauvegarde temporaire du fichier uploadé pour load_model
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
                tmp.write(model_selection.read())
                tmp_path = tmp.name
            model = load_model(tmp_path, device=str(device))
            os.unlink(tmp_path)
            model.eval()
            model.to(device)

            transform_e = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            st.write(f"Processing {len(image_paths)} images...")

            class ImageDataset(Dataset):
                def __init__(self, image_paths, transform):
                    self.image_paths = image_paths
                    self.transform = transform
                
                def __len__(self):
                    return len(self.image_paths)
                
                def __getitem__(self, idx):
                    img_path = self.image_paths[idx]
                    img = Image.open(img_path).convert("RGB")
                    img = self.transform(img)
                    img_name = os.path.basename(img_path)
                    return img, img_name

            batch_size_embed = 8
            dataset = ImageDataset(image_paths, transform_e)
            dataloader = DataLoader(dataset, batch_size=batch_size_embed, shuffle=False, num_workers=0, pin_memory=True)

            embeddings = []
            progress_bar = st.progress(0)
            total_batches = len(dataloader)

            with torch.no_grad():
                for batch_idx, (images, img_names) in enumerate(dataloader):
                    images = images.to(device)
                    batch_embeddings = model.encode_backbone(images).cpu().numpy()
                    for img_name, embedding in zip(img_names, batch_embeddings):
                        embeddings.append([img_name] + embedding.tolist())
                    progress_bar.progress((batch_idx + 1) / total_batches)

            df = pd.DataFrame(embeddings)
            df.to_csv(embedding_save_path, index=False, sep=";")
            
            # Extract filenames and feature arrays from the embeddings list
            filenames = [row[0] for row in embeddings]
            feats_array = np.array([row[1:] for row in embeddings], dtype=np.float32)
            
            # Save ref_feats.npy (shape: N x feature_dim)
            feats_save_path = os.path.join(os.path.dirname(embedding_save_path) or ".", "ref_feats.npy")
            np.save(feats_save_path, feats_array)
            
            # Save ref_labels.npy (shape: N,) — derived from the parent subfolder name of each image
            labels_array = np.array([
                os.path.basename(os.path.dirname(p)) for p in image_paths
            ])
            labels_save_path = os.path.join(os.path.dirname(embedding_save_path) or ".", "ref_labels.npy")
            np.save(labels_save_path, labels_array)
            
            st.success(f"Embeddings saved to {embedding_save_path}")
            st.success(f"ref_feats.npy saved ({feats_array.shape[0]} images × {feats_array.shape[1]} features) → {feats_save_path}")
            st.success(f"ref_labels.npy saved ({len(labels_array)} labels) → {labels_save_path}")
            progress_bar.empty()
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    
# === Validation Page ===
elif page == "Validation":
    st.header("SimCLR Validation")

    model_path = st.text_input("Enter Path to Trained Model", value="simclr_model.pth")
    dataset_path = st.text_input("Enter Validation Dataset Path", value="")
    batch_size_validation = st.slider("Batch Size for validation", 16, 128, 32)
    start_validation = st.button("Start SimCLR Validation")

    def train_linear_classifier(model, dataloaders, dataset_sizes, num_epochs=50, lr=0.001):
        since = time.time()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.projection_head.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(num_epochs*0.6), int(num_epochs*0.8)], gamma=0.1)
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        logtxtbox = st.empty()
        logtxtbox2 = st.empty()
        train_loss_history = []
        val_loss_history = []
        plot_spot = st.empty()

        fig, ax = plt.subplots()
        train_loss_line, = ax.plot([], [], marker="o", label="Train loss")
        val_loss_line, = ax.plot([], [], marker="x", label="Validation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Progress")
        ax.legend()

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            for phase in ['train', 'validation']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                running_loss = 0.0
                running_corrects = 0
                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                if phase=="train" :
                    logtxtbox.write(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                    train_loss_history.append(epoch_loss)
                elif phase=="validation" :
                    logtxtbox2.write(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                    val_loss_history.append(epoch_loss)
                train_loss_line.set_data(range(1, len(train_loss_history) + 1), train_loss_history)
                val_loss_line.set_data(range(1, len(val_loss_history) + 1), val_loss_history)
                ax.relim()
                ax.autoscale_view()
                with plot_spot:
                    st.pyplot(fig)
                if phase == 'validation' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        logtxtbox.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s, Best Validation Accuracy: {best_acc:.4f}')
        model.load_state_dict(best_model_wts)
        return model

    def evaluate_classifier(model, dataloader, class_names):
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        correct_pred = {classname: 0 for classname in class_names}
        total_pred = {classname: 0 for classname in class_names}

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                for label, prediction in zip(labels, preds):
                    if label == prediction:
                        correct_pred[class_names[label]] += 1
                    total_pred[class_names[label]] += 1

        accuracy = 100 * correct / total
        f1 = f1_score(all_labels, all_preds, average='weighted')
        kappa = cohen_kappa_score(all_labels, all_preds)
        for classname in class_names:
            if total_pred[classname] > 0:
                class_accuracy = 100 * correct_pred[classname] / total_pred[classname]
                print(f'Accuracy for class {classname}: {class_accuracy:.1f}%')
        return accuracy, f1, kappa

    if start_validation and model_path and dataset_path:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            transform_val = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            dataset = datasets.ImageFolder(root=dataset_path, transform=transform_val)
            class_names = dataset.classes
            model = load_model(model_path, device=str(device))
            feat_dim = model.feature_dim  # 2048 pour resnet50, 512 pour resnet18/34
            # Geler le backbone, remplacer la tête par un classifieur linéaire
            for param in model.encoder.parameters():
                param.requires_grad = False
            model.projection_head = nn.Sequential(
                nn.Dropout(),
                nn.Linear(feat_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, len(class_names))
            )
            model = model.to(device)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size_validation, shuffle = True, num_workers=4)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size_validation, shuffle=True, num_workers=4)
            dataloaders = {'train': train_dataloader, 'validation': val_dataloader}
            dataset_sizes = {'train': len(train_dataset), 'validation': len(val_dataset)}
            st.write("Training linear classifier...")
            model = train_linear_classifier(model, dataloaders, dataset_sizes, num_epochs=50, lr=0.001)
            st.write("Evaluating classifier...")
            accuracy, f1, kappa = evaluate_classifier(model, val_dataloader, class_names)
            st.write(f"Validation Accuracy: {accuracy:.2f}%")
            st.write(f"Validation F1 Score: {f1:.4f}")
            st.write(f"Validation Cohen's Kappa: {kappa:.4f}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# === GradCAM Page (unchanged) ===
if page == "GradCAM":
    st.header("🔍 Grad-CAM Visualization")

    gradcam_mode = st.radio("Select Input Mode", ["Single Image", "Folder"], horizontal=True)
    if gradcam_mode == "Single Image":
        gradcam_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    else:
        gradcam_folder = st.text_input("Select a folder of images")

    output_folder = st.text_input("Output folder to save Grad-CAM images", "gradcam_results")
    model_path = st.text_input("Enter Path to Trained Model", value="simclr_model.pth")
    run_gradcam = st.button("Run Grad-CAM")

    if run_gradcam:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.write(f"Using device: `{device}`")
        model = load_model(model_path, device=str(device))
        model.to(device)
        model.eval()
        # Hook layer (last conv layer)
        target_layers = [model.encoder.layer3]
        cam = GradCAM(model=model, target_layers=target_layers)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        os.makedirs(output_folder, exist_ok=True)
        def process_image(image: Image.Image, filename: str):
            resized_image = image.resize((224, 224))
            rgb_img = np.array(resized_image).astype(np.float32) / 255
            input_tensor = transform(image).unsqueeze(0).to(device)
            grayscale_cam = cam(input_tensor=input_tensor, targets = None)[0]
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            result_img = Image.fromarray(visualization)
            result_img.save(os.path.join(output_folder, filename))
            return result_img
        if gradcam_mode == "Single Image" and gradcam_img is not None:
            image = Image.open(gradcam_img).convert("RGB")
            result = process_image(image, "gradcam_result.png")
            st.image(result, caption="Grad-CAM Result")
        elif gradcam_mode == "Folder" and gradcam_folder:
            files = []
            for root, dirs, filenames in os.walk(gradcam_folder):
                for f in filenames:
                    if f.lower().endswith((".jpg", ".png", ".jpeg")):
                        files.append(os.path.join(root, f))
            prog = st.progress(0.0, text="Running Grad-CAM...")
            for i, fname in enumerate(files):
                image = Image.open(fname).convert("RGB")
                result = process_image(image, f"{os.path.splitext(os.path.split(fname)[-1])[0]}_gradcam.png")
                prog.progress((i + 1) / len(files), text=f"Processing {fname}")
            st.success(f"Saved Grad-CAM for {len(files)} images.")
            
            
if page == "Visualization":
            
    import streamlit as st
    import pandas as pd
    import numpy as np
    import os
    from PIL import Image
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import umap
    from distinctipy import distinctipy
    from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import accuracy_score
    import plotly.graph_objects as go
    
    # --- Hit-rate@k (overall and per class)
    def hit_rate_at_k(X, y, k=5):
        """
        X: feature matrix (n_samples, n_features)
        y: labels (n_samples,)
        k: number of neighbors to consider (excluding self)
        Returns:
            overall_hit_rate: float
            per_class_hit_rate: dict {label: hit-rate}
        """
        nn = NearestNeighbors(n_neighbors=k+1)  # +1 because first neighbor is self
        nn.fit(X)
        dists, indices = nn.kneighbors(X)
    
        # Exclude self (first column)
        indices = indices[:, 1:]
    
        hits = np.zeros(len(y), dtype=int)
        for i, neighbors in enumerate(indices):
            neighbor_labels = y[neighbors]
            hits[i] = int(y[i] in neighbor_labels)
    
        overall_hit_rate = hits.mean()
    
        per_class_hit_rate = {lab: hits[y == lab].mean() for lab in np.unique(y)}
    
        return overall_hit_rate, per_class_hit_rate


    st.title("Embeddings Visualization & k-NN Evaluation (Interactive)")

    embeddings_csv = st.text_input(
        "Path to embeddings CSV:",
        r"G:\Mon Drive\CDD MNHN\embeddings.csv"
    )

    # --- How labels are obtained ---
    label_source = st.radio(
        "How should labels be loaded?",
        ["CSV file with labels", "Dataset folder structure"]
    )

    labels_csv = None
    dataset_folder = None
    label_column_name = None

    if label_source == "CSV file with labels":
        labels_csv = st.text_input(
            "Path to labels CSV:",
            r"G:\Mon Drive\CDD MNHN\labels_clean.csv"
        )

        # Allow choosing the label column
        if os.path.exists(labels_csv):
            tmp_df = pd.read_csv(labels_csv, sep=";")
            label_column_name = st.selectbox(
                "Which column contains the labels?",
                tmp_df.columns.tolist(),
                index=tmp_df.columns.tolist().index("label") if "label" in tmp_df.columns else 0
            )

    else:
        dataset_folder = st.text_input(
            "Dataset root folder (contains class-name subfolders):",
            r"C:\Users\agaca\Mon Drive\CDD MNHN\dataset"
        )

    image_folder = st.text_input(
        "Folder with images:",
        r"G:\Mon Drive\CDD MNHN\rembg\resized_withoutbg"
    )

    method = st.selectbox("Dimensionality reduction method:", ["pca", "tsne", "umap"])
    
    k = st.number_input("k for k-NN classifier (CV):", value=5, min_value=1)
    k_nn = st.number_input("k for Hit-Rate:", value=5, min_value=1)

    # NEW — this now applies ONLY to kNN eval, not to scatter plot
    min_class_size = st.number_input(
        "Minimum number of images per class:",
        value=1, min_value=1
    )

    run_button = st.button("Run Analysis")

    if run_button:
        # ------------------------------------------
        # Load embeddings
        # ------------------------------------------
        emb = pd.read_csv(embeddings_csv, sep=";")
        emb["id"] = (
            emb.iloc[:, 0]
            .str.replace(".jpg", "", regex=False)
            .str.replace(".png", "", regex=False)
        )

        # ------------------------------------------
        # OPTION 1 — LABELS FROM CSV
        # ------------------------------------------
        if label_source == "CSV file with labels":
            labels = pd.read_csv(labels_csv, sep=";")
            labels["id"] = labels["id"].astype(str)

            merged = emb.merge(labels, on="id", how="left")

            merged["label"] = merged[label_column_name]

        # ------------------------------------------
        # OPTION 2 — FOLDER STRUCTURE LABELS
        # ------------------------------------------
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
                        os.path.exists(os.path.join(p, base + ".jpg")) or
                        os.path.exists(os.path.join(p, base + ".png"))
                    ):
                        detected = cls
                        break

                label_list.append(detected)

            emb["label"] = label_list
            merged = emb

        # Remove rows without labels
        merged = merged.dropna(subset=["label"])

        # ============================================================
        # DO NOT filter small classes for the scatter plot
        # ============================================================
        X_all = merged.iloc[:, 1:2049].values
        X_all = np.nan_to_num(X_all)
        y_all = merged["label"].values

        # --- Dimensionality reduction ---
        if method == "pca":
            reducer = PCA(n_components=2)
        elif method == "tsne":
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        else:
            reducer = umap.UMAP(n_components=2, random_state=42)

        proj = reducer.fit_transform(X_all)
        merged["x"], merged["y"] = proj[:, 0], proj[:, 1]

        # ------------------------------------------
        # Colors
        # ------------------------------------------
        unique_labels = merged["label"].unique()
        colors = distinctipy.get_colors(len(unique_labels))
        color_dict = {
            lab: f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"
            for lab, c in zip(unique_labels, colors)
        }
        merged["color"] = merged["label"].map(color_dict)

        # ------------------------------------------
        # PLOT
        # ------------------------------------------
        fig = go.Figure()

        for lbl in unique_labels:
            s = merged[merged["label"] == lbl]
            fig.add_trace(go.Scatter(
                x=s["x"],
                y=s["y"],
                mode='markers',
                marker=dict(color=color_dict[lbl], size=20),
                name=lbl,
                text=s["id"],
                hovertemplate="<b>%{text}</b><br>Label: "+lbl+"<extra></extra>"
            ))

        # Overlay images
        for _, r in merged.iterrows():
            fp = os.path.join(image_folder, r.iloc[0])
            fp2 = fp.replace(".jpg", ".png")
            final = fp2 if os.path.exists(fp2) else fp

            if os.path.exists(final):
                img = Image.open(final).convert("RGBA")
                fig.add_layout_image(dict(
                    source=img,
                    x=r["x"],
                    y=r["y"],
                    xref="x",
                    yref="y",
                    xanchor="center",
                    yanchor="middle",
                    sizex=1.5,
                    sizey=1.5,
                    sizing="contain",
                    layer="above",
                    opacity=1
                ))

        fig.update_layout(
            title=f"{method.upper()} Morphospace",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            hovermode="closest",
            width=1000,
            height=800
        )

        st.plotly_chart(fig, use_container_width=True)
        
        

        # ============================================================
        # k-NN evaluation ONLY on classes >= min_class_size
        # ============================================================
        df_eval = merged.copy()

        if min_class_size > 1:
            counts = df_eval["label"].value_counts()
            keep = counts[counts >= min_class_size].index
            df_eval = df_eval[df_eval["label"].isin(keep)]

        # Continue only if we have >1 class
        if df_eval["label"].nunique() >= 2:

            X = df_eval.iloc[:, 1:2049].values
            y = df_eval["label"].values
            
            # --- Hit-rate@k
            overall_hr, per_class_hr = hit_rate_at_k(X, y, k=k_nn)
            st.write(f"Hit-rate @ {k_nn} nearest neighbors (overall): {overall_hr:.3f}")
            st.dataframe(
                pd.DataFrame.from_dict(per_class_hr, orient="index", columns=[f"Hit-rate@{k_nn}"])
            )

            # kNN CV
            knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            y_pred = cross_val_predict(knn, X, y, cv=cv)

            acc = accuracy_score(y, y_pred)
            st.write(f"Overall k-NN accuracy (5-fold CV): {acc:.3f}")

            per_label = {
                lab: accuracy_score(y[y == lab], y_pred[y == lab])
                for lab in np.unique(y)
            }
            st.dataframe(
                pd.DataFrame.from_dict(per_label, orient="index", columns=["kNN_accuracy"])
            )

            # # Nearest neighbor cosine
            # nn = NearestNeighbors(n_neighbors=k_nn + 1)#, metric="cosine")
            # nn.fit(X)
            # dist, idx = nn.kneighbors(X)
            # nearest = idx[:, 1]
            # same = (y[nearest] == y)

            # st.write(f"Overall nearest-neighbor accuracy: {same.mean():.3f}")

            # per_label_nn = {lab: same[y == lab].mean() for lab in np.unique(y)}
            # st.dataframe(
            #     pd.DataFrame.from_dict(per_label_nn, orient="index", columns=["NN_accuracy"])
            # )
            
            

        else:
            st.warning("Not enough classes with sufficient size for k-NN evaluation.")
