import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import glob
from torch.utils.data import WeightedRandomSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
import time
import copy
from typing import Optional, Tuple, Union
import random
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
# from pytorch_grad_cam.utils.model_targets import EigenCAMTarget
from pytorch_grad_cam.utils.image import preprocess_image
from sklearn.manifold import TSNE
import plotly.express as px
from PIL import Image
import base64
from io import BytesIO
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import base64
import io
from io import BytesIO
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# === Encoder Model ===
class Encoder(nn.Module):
    def __init__(self, base_model=models.resnet50, out_dim=128):
        super().__init__()
        self.encoder = base_model(pretrained=True)
        self.encoder.fc = nn.Identity()  # Remove classification head
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        return self.projection_head(features)

# === NT-Xent Loss Function ===
def nt_xent_loss(z1, z2, temperature=0.5):
    z = torch.cat((z1, z2), dim=0)
    z = nn.functional.normalize(z, dim=1)
    similarity = torch.mm(z, z.T) / temperature
    labels = torch.arange(z.size(0), device=z.device)
    labels = (labels + z.size(0) // 2) % z.size(0)  # Positive pairs
    loss = nn.CrossEntropyLoss()(similarity, labels)
    return loss

# === SimCLR Transformations ===
class SimCLRTransform:
    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 0.5,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.2, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        normalize: dict = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
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

    def __call__(self, x):
        return self.transform(x), self.transform(x)


# === Streamlit UI ===
st.title("SimCLR Training GUI")

# Navigation
page = st.sidebar.radio("Select a page:", ["Training", "Generate Embeddings", "Validation", "GradCAM"])

# === Training Page ===
if page == "Training":
    st.header("SimCLR Training")

    # User Inputs
    dataset_path = st.text_input("Enter Dataset Path", value="")
    save_model_path = st.text_input("Enter Model Save Path", value="simclr_model.pth")
    batch_size = st.slider("Batch Size", 16, 128, 32)
    epochs = st.slider("Epochs", 1, 300, 10)
    learning_rate = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")

    # Image Transformation Probabilities
    input_size = st.number_input("Input Size", min_value=64, max_value=512, value=224, step=8)

    min_scale = st.slider("Minimum Scale for RandomResizedCrop", 0.01, 1.0, 0.08, 0.01)
    
    cj_col = st.columns(2)
    with cj_col[0]:
        cj_prob = st.slider("Color Jitter Probability", 0.0, 1.0, 0.8, 0.01)
    with cj_col[1]:
        cj_strength = st.slider("Color Jitter Strength", 0.0, 1.0, 0.5, 0.01)
    
    flip_col = st.columns(2)
    with flip_col[0]:
        hf_prob = st.slider("Horizontal Flip Probability", 0.0, 1.0, 0.5, 0.01)
    with flip_col[1]:
        vf_prob = st.slider("Vertical Flip Probability", 0.0, 1.0, 0.0, 0.01)
    
    gray_blur_col = st.columns(2)
    with gray_blur_col[0]:
        random_grayscale_prob = st.slider("Random Grayscale Probability", 0.0, 1.0, 0.2, 0.01)
    with gray_blur_col[1]:
        gaussian_blur_prob = st.slider("Gaussian Blur Probability", 0.0, 1.0, 0.5, 0.01)
    
    blur_col = st.columns(2)
    with blur_col[0]:
        sigmas_min = st.number_input("Min Sigma for Gaussian Blur", 0.01, 5.0, 0.2, 0.01)
    with blur_col[1]:
        sigmas_max = st.number_input("Max Sigma for Gaussian Blur", 0.1, 5.0, 2.0, 0.01)
    
    rotation = st.checkbox("Enable Random Rotation", value=False)
    rr_prob = st.slider("Random Rotation Probability", 0.0, 1.0, 0.0, 0.01) if rotation else 0.0
    rr_degrees = st.slider("Rotation Degrees Range", 0, 180, 45, 1) if rotation else None
    
    normalize = {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }

    start_training = st.button("Start Training")

    # === Training Function ===
    def train_simclr_gui(model, dataloader, epochs, lr, device, transform):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.to(device)
        loss_history = []
        
        plot_spot = st.empty()  # holding the spot for the graph
        
        # creating a placeholder for the fixed sized textbox
        logtxtbox = st.empty()

        fig, ax = plt.subplots()
        loss_line, = ax.plot([], [], marker="o", label="Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Progress")
        ax.legend()

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

            for (x1, x2), _ in progress_bar:
                x1, x2 = x1.to(device), x2.to(device)
                z1, z2 = model(x1), model(x2)
                loss = nt_xent_loss(z1, z2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = total_loss / len(dataloader)
            loss_history.append(avg_loss)

            loss_line.set_data(range(1, len(loss_history) + 1), loss_history)
            ax.relim()  # Recalculate axis limits
            ax.autoscale_view()  # Autoscale the view to fit the new data

            logtxtbox.write(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            with plot_spot:
                st.pyplot(fig)

    # === Start Training When Button Clicked ===
    if start_training and dataset_path:
        try:
            st.write("Loading dataset...")
            
            # Apply selected transformations with probabilities
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
            normalize=normalize
        )


            dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
            
            # Count the occurrences of each class
            class_counts = [0] * len(dataset.classes)
            for _, label in dataset.samples:
                class_counts[label] += 1
            
            # Compute class weights (inverse of frequency)
            class_weights = [1.0 / count for count in class_counts]
            
            # Assign weights to each sample based on its class
            weights = [class_weights[label] for _, label in dataset.samples]
            
            # Create a WeightedRandomSampler
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

            dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
            st.write(f"Dataset Loaded! {len(dataset)} images found.")
            model = Encoder()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            train_simclr_gui(model, dataloader, epochs, learning_rate, device, transform)
            st.write("Training Complete! Saving model...")
            torch.save(model.state_dict(), save_model_path)
            st.success(f"Model saved to {save_model_path}")
        except Exception as e:
            st.error(f"Error: {str(e)}")


# === Embeddings Page ===
elif page == "Generate Embeddings":
    st.header("Generate Embeddings")
    embeddings_folder = st.text_input("Enter Folder Path for Embeddings", value="")
    model_selection = st.file_uploader("Upload Model for Embeddings", type=["pth"])
    embedding_save_path = st.text_input("Enter Path to Save Embeddings CSV", value="embeddings.csv")
    # image_paths = glob.glob(os.path.join(embeddings_folder, "*.jpg"))
    image_paths = []
    for root, dirs, filenames in os.walk(embeddings_folder):
        for f in filenames:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(root, f))  # This gives you the full path to the file
    start_embedding = st.button("Generate Embeddings")

    if start_embedding and embeddings_folder and model_selection:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Encoder()
            model.load_state_dict(torch.load(model_selection, map_location=device))
            model.eval()
            model.to(device)
    
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
    
            # Collect image paths
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
                    img_name = os.path.basename(img_path)  # Extract only the filename
                    return img, img_name
    
            # Create dataset and dataloader for batch processing
            batch_size = 1  # You can adjust this based on your GPU memory
            dataset = ImageDataset(image_paths, transform)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
            embeddings = []
            progress_bar = st.progress(0)  # Streamlit progress bar
            total_batches = len(dataloader)
    
            with torch.no_grad():
                for batch_idx, (images, img_names) in enumerate(dataloader):
                    images = images.to(device)
                    batch_embeddings = model.encoder(images).cpu().numpy()
                    for img_name, embedding in zip(img_names, batch_embeddings):
                        embeddings.append([img_name] + embedding.tolist())
    
                    # Update progress bar
                    progress_bar.progress((batch_idx + 1) / total_batches)
    
            df = pd.DataFrame(embeddings)
    
            # Save CSV with semicolon delimiter
            df.to_csv(embedding_save_path, index=False, sep=";")
    
            st.success(f"Embeddings saved to {embedding_save_path}")
            progress_bar.empty()  # Clear the progress bar
    
        except Exception as e:
            st.error(f"Error: {str(e)}")

    # # === t-SNE Visualization ===
    
    # st.subheader("🔍 Visualize Embeddings with t-SNE")
    
    # # Add a user option for selecting the image format
    # image_format = st.selectbox("Select Image Format", ["jpg", "png"], index=1)
    
    # tsne_button = st.button("Run t-SNE and Show Image Scatterplot")
    # perplexity = st.slider("t-SNE Perplexity", min_value=5, max_value=50, value=30, step=1)
    # thumbnail_size = st.slider("Thumbnail Size (px)", 10, 50, 30)
    
    # if tsne_button:
    #     try:
    #         df = pd.read_csv(embedding_save_path, sep=";")
    #         filenames = df.iloc[:, 0].tolist()
    #         features = df.iloc[:, 1:].values
    
    #         # Run t-SNE
    #         st.write("Running t-SNE...")
    #         tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    #         tsne_result = tsne.fit_transform(features)
    
    #         # Generate plot with image thumbnails
    #         fig, ax = plt.subplots(figsize=(10, 10))
    #         ax.set_title("t-SNE Image Scatterplot")
    
    #         for i, (x, y) in enumerate(tsne_result):
    #             try:
    #                 # Explicitly replace the file extension with the selected image format
    #                 img_path = os.path.join(embeddings_folder, f"{os.path.splitext(filenames[i])[0]}.{image_format}")
    #                 img = Image.open(img_path).convert("RGB")
                    
    #                 # Resize image to fit within the plot constraints (e.g., 224x224)
    #                 img = img.resize((64, 64))  # Resize image to 224x224 (or another appropriate size)
    #                 img.thumbnail((thumbnail_size, thumbnail_size))
    #                 img_arr = np.asarray(img)
    
    #                 imagebox = OffsetImage(img_arr, zoom=1)
    #                 ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    #                 ax.add_artist(ab)
    
    #             except Exception as e:
    #                 print(f"Could not load image {filenames[i]}: {e}")
    
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    
    #         st.pyplot(fig)
    
    #         # Save PNG button
    #         png_output = BytesIO()
    #         fig.savefig(png_output, format="png", bbox_inches='tight')
    #         b64 = base64.b64encode(png_output.getvalue()).decode()
    #         href = f'<a href="data:image/png;base64,{b64}" download="tsne_plot.png">📥 Download Plot as PNG</a>'
    #         st.markdown(href, unsafe_allow_html=True)
    
    #     except Exception as e:
    #         st.error(f"t-SNE visualization failed: {e}")
    






# === Validation Page ===
elif page == "Validation":
    st.header("SimCLR Validation")

    model_path = st.text_input("Enter Path to Trained Model", value="simclr_model.pth")
    dataset_path = st.text_input("Enter Validation Dataset Path", value="")
    batch_size_validation = st.slider("Batch Size for validation", 16, 128, 32)

    start_validation = st.button("Start SimCLR Validation")
    
    # === Model Training Function ===
    def train_linear_classifier(model, dataloaders, dataset_sizes, num_epochs=50, lr=0.001):
        """Train a linear classifier on top of a frozen encoder."""
        since = time.time()
    
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.projection_head.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(num_epochs*0.6), int(num_epochs*0.8)], gamma=0.1)
    
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        
        # creating a placeholder for the fixed sized textbox
        logtxtbox = st.empty()
        logtxtbox2 = st.empty()
        
        train_loss_history = []
        val_loss_history = []
        
        plot_spot = st.empty()  # holding the spot for the graph
        
        
        fig, ax = plt.subplots()
        train_loss_line, = ax.plot([], [], marker="o", label="Train loss")
        val_loss_line, = ax.plot([], [], marker="x", label="Validation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Progress")
        ax.legend()
    
        # Training loop
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)
    
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
    
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                if phase=="train" :
                    logtxtbox.write(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                elif phase=="validation" :
                    logtxtbox2.write(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                if phase=="train" :
                    train_loss_history.append(epoch_loss)
                elif phase=="validation" :
                    val_loss_history.append(epoch_loss)

                train_loss_line.set_data(range(1, len(train_loss_history) + 1), train_loss_history)
                val_loss_line.set_data(range(1, len(val_loss_history) + 1), val_loss_history)
                ax.relim()  # Recalculate axis limits
                ax.autoscale_view()  # Autoscale the view to fit the new data

                with plot_spot:
                    st.pyplot(fig)
                    
                # Save best model
                if phase == 'validation' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
    
        time_elapsed = time.time() - since
        logtxtbox.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s, Best Validation Accuracy: {best_acc:.4f}')
        # print(f'Best Validation Accuracy: {best_acc:.4f}')
    
        model.load_state_dict(best_model_wts)
        return model
    
    # === Model Evaluation Function ===
    def evaluate_classifier(model, dataloader, class_names):
        """Evaluate the trained classifier and compute accuracy, F1-score, and Cohen’s kappa."""
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
    
        print(f'Overall Accuracy: {accuracy:.2f}%')
        print(f'F1 Score: {f1:.4f}')
        print(f'Cohen’s Kappa: {kappa:.4f}')
    
        for classname in class_names:
            if total_pred[classname] > 0:
                class_accuracy = 100 * correct_pred[classname] / total_pred[classname]
                print(f'Accuracy for class {classname}: {class_accuracy:.1f}%')
    
        return accuracy, f1, kappa
    
    # === Load Model and Dataset in Streamlit ===
    if start_validation and model_path and dataset_path:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Define dataset transformations
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
            dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
            class_names = dataset.classes
    
            # Load trained SimCLR model
            model = Encoder(base_model=models.resnet50)
            model.load_state_dict(torch.load(model_path, map_location=device))
    
            # Remove projection head and add a linear classifier
            model.projection_head = nn.Identity()
            for param in model.encoder.parameters():
                param.requires_grad = False
    
            model.projection_head = nn.Sequential(
                nn.Dropout(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, len(class_names))
            )
            
            model = model.to(device)
            
            # Ensure the classifier is trainable
            for param in model.projection_head.parameters():
                param.requires_grad = True  # ✅ Explicitly enable gradients
    
    
            # Split dataset (80% train, 20% validation)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            # # Count the occurrences of each class
            # class_counts = [0] * len(dataset.classes)
            # for _, label in dataset.samples:
            #     class_counts[label] += 1
            
            # # Compute class weights (inverse of frequency)
            # class_weights = [1.0 / count for count in class_counts]
            
            # # Assign weights to each sample based on its class
            # weights = [class_weights[label] for _, label in dataset.samples]
            
            # # Create a WeightedRandomSampler
            # sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
            # Create dataloaders
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size_validation, shuffle = True, num_workers=4)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size_validation, shuffle=True, num_workers=4)
    
            dataloaders = {'train': train_dataloader, 'validation': val_dataloader}
            dataset_sizes = {'train': len(train_dataset), 'validation': len(val_dataset)}
    
            # Train linear classifier
            st.write("Training linear classifier...")
            model = train_linear_classifier(model, dataloaders, dataset_sizes, num_epochs=50, lr=0.001)
    
            # Evaluate the trained classifier
            st.write("Evaluating classifier...")
            accuracy, f1, kappa = evaluate_classifier(model, val_dataloader, class_names)
    
            # Display results
            st.write(f"Validation Accuracy: {accuracy:.2f}%")
            st.write(f"Validation F1 Score: {f1:.4f}")
            st.write(f"Validation Cohen's Kappa: {kappa:.4f}")
    
        except Exception as e:
            st.error(f"Error: {str(e)}")



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

        # Load model
        model = Encoder()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Hook layer (last conv layer)
        target_layers = [model.encoder.layer3]

        cam = GradCAM(model=model, target_layers=target_layers)#, use_cuda=torch.cuda.is_available())

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        os.makedirs(output_folder, exist_ok=True)

        def process_image(image: Image.Image, filename: str):
            resized_image = image.resize((224, 224))  # Resize to match input tensor
            rgb_img = np.array(resized_image).astype(np.float32) / 255  # Now it's (224, 224, 3)
            input_tensor = transform(image).unsqueeze(0).to(device)

            grayscale_cam = cam(input_tensor=input_tensor, targets = None)[0]
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            result_img = Image.fromarray(visualization)
            result_img.save(os.path.join(output_folder, filename))

            return result_img

        if gradcam_mode == "Single Image" and gradcam_img is not None:
            image = Image.open(gradcam_img).convert("RGB")
            result = process_image(image, "gradcam_result.png")
            st.image(result, caption="Grad-CAM Result", use_container_width=True)

        elif gradcam_mode == "Folder" and gradcam_folder:
            files = []
            for root, dirs, filenames in os.walk(gradcam_folder):
                for f in filenames:
                    if f.lower().endswith((".jpg", ".png", ".jpeg")):
                        files.append(os.path.join(root, f))  # This gives you the full path to the file
            prog = st.progress(0.0, text="Running Grad-CAM...")

            for i, fname in enumerate(files):
                img_path = os.path.join(fname)
                image = Image.open(img_path).convert("RGB")
                result = process_image(image, f"{os.path.splitext(os.path.split(fname)[-1])[0]}_gradcam.png")
                prog.progress((i + 1) / len(files), text=f"Processing {fname}")

            st.success(f"Saved Grad-CAM for {len(files)} images.")