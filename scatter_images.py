import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os
from pathlib import Path
import tqdm
import cv2  # OpenCV for faster resizing
import argparse
from sklearn.decomposition import PCA
import plotly.graph_objects as go

def convert_white_to_transparent_cv(img: np.ndarray, threshold: int = 240) -> np.ndarray:
    """
    Converts white (or near-white) background to transparent in an OpenCV image.

    Args:
        img: OpenCV image as a NumPy array (BGR or BGRA).
        threshold: Intensity threshold to consider a pixel as white.

    Returns:
        OpenCV image with alpha channel where white pixels are transparent.
    """
    # Ensure the image has 4 channels (BGRA)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # Create mask of white pixels (or nearly white)
    white_mask = np.all(img[:, :, :3] > threshold, axis=2)

    # Set alpha to 0 where mask is True (white), else 255
    img[:, :, 3] = np.where(white_mask, 0, 255)

    return img

parser = argparse.ArgumentParser(description="t-SNE and PCA Scatterplot with Thumbnails")
parser.add_argument("--csv", default="embeddings.csv", required=False, help="Path to the CSV file with embeddings")
parser.add_argument("--img_folder", default="data", required=False, help="Path to the folder with images")
parser.add_argument("--format", default="png", help="Image format: jpg or png (default: png)")
parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity (default: 30)")
parser.add_argument("--thumb_size", type=int, default=100, help="Thumbnail size in pixels (default: 100)")
args = parser.parse_args()

# Assign to variables
embedding_save_path = args.csv
embeddings_folder = args.img_folder
image_format = args.format.lower()
perplexity = args.perplexity
thumbnail_size = args.thumb_size

# Load embeddings
df = pd.read_csv(embedding_save_path, sep=";")
filenames = df.iloc[:, 0].tolist()
features = df.iloc[:, 1:].values

# Standardize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Run t-SNE
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
tsne_result = tsne.fit_transform(features)

# Apply PCA
pca = PCA()
pca_result = pca.fit_transform(features)

# Keep components explaining more than 1% variance
explained_variance = pca.explained_variance_ratio_
num_components = np.sum(explained_variance > 0.01)

# Prepare PCA result with retained components
pca_result_reduced = pca_result[:, :num_components]

# Create Plotly figure for t-SNE (PC1 vs PC2)
fig_tsne = px.scatter(x=tsne_result[:, 0], y=tsne_result[:, 1])

# Prepare a list to store images and coordinates
images_data_tsne = []

# Get x and y ranges for size adjustment
x_range = tsne_result[:, 0].max() - tsne_result[:, 0].min()
y_range = tsne_result[:, 1].max() - tsne_result[:, 1].min()
size_fraction = 0.1  # Adjust this to control thumbnail visibility
sizex = x_range * size_fraction
sizey = y_range * size_fraction

# Create the tqdm progress bar to update in the console
progress_bar = tqdm.tqdm(total=len(filenames), desc="Processing Images (t-SNE)", ncols=100)

# Loop through the images and prepare the data for t-SNE
for x, y, filename in zip(fig_tsne.data[0].x, fig_tsne.data[0].y, filenames):
    try:
        search_pattern = f"{os.path.splitext(filename)[0]}.{image_format}"
        matches = list(Path(embeddings_folder).rglob(search_pattern))

        if matches:
            img_path = str(matches[0])
        else:
            img_path = None

        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGBA")
            img_cv = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            if image_format == "jpg":
                img_cv = convert_white_to_transparent_cv(img_cv)

            img_resized = cv2.resize(img_cv, (thumbnail_size, thumbnail_size))
            img_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGRA2RGBA))

            images_data_tsne.append({
                'x': x,
                'y': y,
                'source': img_pil,
                'sizex': sizex,
                'sizey': sizey,
                'xanchor': "center",
                'yanchor': "middle"
            })

    except Exception as e:
        print(f"Could not load image {filename}: {e}")
    
    progress_bar.update(1)

progress_bar.close()

# Now, update the t-SNE plot with all images in one go
fig_tsne.update_layout(
    images=[{
        'x': data['x'],
        'y': data['y'],
        'source': data['source'],
        'xref': 'x',
        'yref': 'y',
        'sizex': data['sizex'],
        'sizey': data['sizey'],
        'xanchor': data['xanchor'],
        'yanchor': data['yanchor']
    } for data in images_data_tsne]
)

# Customize the layout for t-SNE
fig_tsne.update_layout(
    title="t-SNE Image Scatterplot",
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False),
    hovermode='closest'
)

# PCA plots

# Get x and y ranges for size adjustment
x_range = pca_result_reduced[:, 0].max() - pca_result_reduced[:, 0].min()
y_range = pca_result_reduced[:, 1].max() - pca_result_reduced[:, 1].min()
size_fraction = 0.1  # Adjust this to control thumbnail visibility
sizex = x_range * size_fraction
sizey = y_range * size_fraction

fig_pca_pc1_pc2 = px.scatter(x=pca_result_reduced[:, 0], y=pca_result_reduced[:, 1])
fig_pca_pc1_pc2.update_layout(
    title=f"PCA PC1 vs PC2 - Explained Variance: {explained_variance[0] * 100:.2f}% vs {explained_variance[1] * 100:.2f}%",
    xaxis=dict(title=f"PC1 ({explained_variance[0] * 100:.2f}% variance)"),
    yaxis=dict(title=f"PC2 ({explained_variance[1] * 100:.2f}% variance)"),
    hovermode='closest'
)

# Add images to PCA plot

# Create the tqdm progress bar to update in the console
progress_bar = tqdm.tqdm(total=len(filenames), desc="Processing Images (PCA axis 1-2)", ncols=100)

images_data_pca = []
for x, y, filename in zip(fig_pca_pc1_pc2.data[0].x, fig_pca_pc1_pc2.data[0].y, filenames):
    try:
        search_pattern = f"{os.path.splitext(filename)[0]}.{image_format}"
        matches = list(Path(embeddings_folder).rglob(search_pattern))

        if matches:
            img_path = str(matches[0])
        else:
            img_path = None

        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGBA")
            img_cv = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            if image_format == "jpg":
                img_cv = convert_white_to_transparent_cv(img_cv)

            img_resized = cv2.resize(img_cv, (thumbnail_size, thumbnail_size))
            img_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGRA2RGBA))
            
        images_data_pca.append({
            'x': x,
            'y': y,
            'source': img_pil,  # Use the same resized image
            'sizex': sizex,
            'sizey': sizey,
            'xanchor': "center",
            'yanchor': "middle"
        })
        
    except Exception as e:
        print(f"Could not load image {filename}: {e}")
    
    progress_bar.update(1)

progress_bar.close()

fig_pca_pc1_pc2.update_layout(
    images=[{
        'x': data['x'],
        'y': data['y'],
        'source': data['source'],
        'xref': 'x',
        'yref': 'y',
        'sizex': data['sizex'],
        'sizey': data['sizey'],
        'xanchor': data['xanchor'],
        'yanchor': data['yanchor']
    } for data in images_data_pca]
)

# Plot PC1 vs PC3 if PC3 explains more than 1% variance
if num_components > 2 and explained_variance[2] > 0.01:
    fig_pca_pc1_pc3 = px.scatter(x=pca_result_reduced[:, 0], y=pca_result_reduced[:, 2])
    
    
    # Create the tqdm progress bar to update in the console
    progress_bar = tqdm.tqdm(total=len(filenames), desc="Processing Images (PCA axis 1-3)", ncols=100)

    images_data_pca13 = []
    for x, y, filename in zip(fig_pca_pc1_pc3.data[0].x, fig_pca_pc1_pc3.data[0].y, filenames):
        try:
            search_pattern = f"{os.path.splitext(filename)[0]}.{image_format}"
            matches = list(Path(embeddings_folder).rglob(search_pattern))

            if matches:
                img_path = str(matches[0])
            else:
                img_path = None

            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGBA")
                img_cv = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                if image_format == "jpg":
                    img_cv = convert_white_to_transparent_cv(img_cv)

                img_resized = cv2.resize(img_cv, (thumbnail_size, thumbnail_size))
                img_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGRA2RGBA))
                
            images_data_pca13.append({
                'x': x,
                'y': y,
                'source': img_pil,  # Use the same resized image
                'sizex': sizex,
                'sizey': sizey,
                'xanchor': "center",
                'yanchor': "middle"
            })
            
        except Exception as e:
            print(f"Could not load image {filename}: {e}")
        
        progress_bar.update(1)

    progress_bar.close()
    
    fig_pca_pc1_pc3.update_layout(
        images=[{
            'x': data['x'],
            'y': data['y'],
            'source': data['source'],
            'xref': 'x',
            'yref': 'y',
            'sizex': data['sizex'],
            'sizey': data['sizey'],
            'xanchor': data['xanchor'],
            'yanchor': data['yanchor']
        } for data in images_data_pca13]
    )
    
    fig_pca_pc1_pc3.update_layout(
        title=f"PCA PC1 vs PC3 - Explained Variance: {explained_variance[0] * 100:.2f}% vs {explained_variance[2] * 100:.2f}%",
        xaxis=dict(title=f"PC1 ({explained_variance[0] * 100:.2f}% variance)"),
        yaxis=dict(title=f"PC3 ({explained_variance[2] * 100:.2f}% variance)"),
        hovermode='closest'
    )

fig_tsne.show()
fig_pca_pc1_pc2.show()
fig_pca_pc1_pc3.show()
    