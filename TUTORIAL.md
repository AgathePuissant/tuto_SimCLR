# 📖 Full Tutorial — SimCLR & SupCon GUI

**Who is this for?** This tutorial is written for anyone curious about visual AI, with no prior machine learning experience. Each section explains *why* we do something before explaining *how* to do it in the interface.

---

## Table of contents

1. [Understanding what we are doing](#1-understanding-what-we-are-doing)
2. [Installation and launch](#2-installation-and-launch)
3. [Preparing your dataset](#3-preparing-your-dataset)
4. [Home page — overview](#4-home-page--overview)
5. [Training a SimCLR model](#5-training-a-simclr-model)
6. [Training a SupCon model](#6-training-a-supcon-model)
7. [Generating embeddings](#7-generating-embeddings)
8. [Validating your model](#8-validating-your-model)
9. [GradCAM visualisation](#9-gradcam-visualisation)
10. [2D embedding visualisation](#10-2d-embedding-visualisation)
11. [Practical tips and troubleshooting](#11-practical-tips-and-troubleshooting)

---

## 1. Understanding what we are doing

### The core idea: learning to "see"

Imagine showing 10,000 photos to someone who has never seen a cat, a dog, or a bird before. Even without being told "that is a cat", they will naturally notice that some images look alike — same fur texture, same body shape, same colours.

That is exactly what **SimCLR** does. We take an image, create two slightly different versions of it (cropped differently, with slightly different colours), and train the model to recognise that those two versions came from the same original photo — even though they look a bit different.

```
Original image
      |
      +---> Version A (different crop, colour shift) ---+
      |                                                  +--> Model learns: "these two are the same thing"
      +---> Version B (different crop, slight blur)  ---+
```

Over time, the model develops a **visual fingerprint** (called an **embedding**) for each image: a vector of numbers that summarises what the image contains. Two similar images will have similar embeddings.

### SimCLR vs SupCon: what is the difference?

| | SimCLR | SupCon |
|---|---|---|
| **Labels required?** | No | Yes |
| **Core principle** | Two views of the same image = positives | Two images of the same class = positives |
| **Best used when** | No labels, or for pre-training | Labels available, best embedding quality |

**SupCon** (Supervised Contrastive Learning) goes one step further: if you have class labels, it uses the knowledge that "this cat photo and that other cat photo are both cats" to produce even better-organised embeddings.

### What you will produce by the end

- A `.pth` model file (containing the learned weights of the neural network)
- An embeddings CSV file (one row per image, columns = embedding dimensions)
- 2D visualisations of your dataset
- GradCAM heatmaps showing where the model "looks" inside your images

---

## 2. Installation and launch

### Step 1 — Install Python

Download Python 3.10 from [python.org](https://www.python.org/downloads/). During installation on Windows, **check the box "Add Python to PATH"**.

To verify Python is correctly installed, open a terminal (PowerShell on Windows, Terminal on Mac/Linux) and type:

```bash
python --version
# You should see: Python 3.10.x
```

### Step 2 — Clone the repository

```bash
git clone https://github.com/your_username/tuto_simclr.git
cd tuto_simclr
```

If you do not have Git installed, click **Code > Download ZIP** on GitHub and extract the archive instead.

### Step 3 — Create a virtual environment

A virtual environment isolates the libraries used by this project from other Python projects on your machine. It avoids version conflicts and is considered best practice.

```bash
# Create the environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac / Linux)
source venv/bin/activate
```

You should see `(venv)` appear at the beginning of your command line. This confirms the environment is active.

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs all required libraries (PyTorch, Streamlit, scikit-learn, etc.). It may take a few minutes.

> **GPU note:** If you have an NVIDIA card and want to use CUDA, replace the `torch` line in `requirements.txt` with the version matching your CUDA toolkit, available at [pytorch.org](https://pytorch.org/get-started/locally/).

### Step 5 — Launch the interface

```bash
streamlit run SimCLR_GUI4.py
```

Your default browser opens automatically at `http://localhost:8501`. If it does not, copy-paste that URL into your browser.

> **To stop the app**, go back to the terminal and press `Ctrl + C`.

---

## 3. Preparing your dataset

### The golden rule: one subfolder = one class

The interface expects your images to be organised like this:

```
my_dataset/
+-- cats/
|   +-- cat_001.jpg
|   +-- cat_002.jpg
|   +-- cat_003.jpg
+-- dogs/
|   +-- dog_001.jpg
|   +-- dog_002.jpg
+-- birds/
    +-- bird_001.jpg
    +-- bird_002.jpg
```

The subfolder name automatically becomes the class label. Accepted formats are `.jpg`, `.jpeg`, and `.png`.

### The provided example images

The `data/` folder in the repository contains a small set of example images organised into subfolders, so you can test the interface immediately without preparing your own dataset. Use `data/` as the dataset path to get started right away.

### Image quality tips

- **Minimum resolution:** 64x64 pixels (the interface automatically resizes to 224x224 by default)
- **Mixed resolutions:** no problem — resizing is handled automatically
- **Complex backgrounds:** no hard restriction, but if your classes are defined by a central object (e.g. insect species), consider removing the background for better results
- **Blurry or corrupted images:** try to remove them — they can harm training

---

## 4. Navigation between pages

Navigation between pages is done through the **left-hand sidebar**.

---

## 5. Training a SimCLR model

Go to **Training** in the sidebar.

### Understanding the parameters

#### Main configuration

**Dataset folder**
The path to your image folder (e.g. `data/` or `C:/my_project/images`). This folder must contain subfolders.

**Model save path**
Where to save the trained model. Example: `runs/simclr_model.pth`. The folder is created automatically if it does not exist.

**Backbone**
The base neural network used to extract visual features. `resnet50` is a solid all-round choice. If you have limited data or a small GPU, try `resnet18`.

| Backbone | Parameters | Speed | Quality |
|---|---|---|---|
| resnet18 | 11M | Very fast | Good |
| resnet34 | 21M | Fast | Good |
| resnet50 | 25M | Medium | Excellent |
| resnet101 | 44M | Slow | Excellent |
| resnet152 | 60M | Slow | Excellent |

**Projection head dimension**
The size of the final embedding produced by the network. `128` is the standard value from the original SimCLR paper. Leave it at `128` unless you have a specific reason to change it.

**Batch Size**
The number of images processed simultaneously. Larger batches give more stable training but require more GPU memory.

| GPU VRAM | Recommended batch size |
|---|---|
| 4 GB | 16-32 |
| 8 GB | 64-128 |
| 16 GB+ | 128-256 |
| CPU only | 8-16 |

**Epochs**
The number of complete passes over the entire dataset. More epochs means more learning, up to a point. Start with 10-20 for initial testing.

**Learning Rate**
The speed of learning. The default `0.001` works well in most cases. Do not change it unless you know what you are doing.

**Number of views per image (N)**
SimCLR creates multiple augmented versions of each image. 2 views is the minimum and most common setting. More views produce richer training but slow things down.

**NT-Xent temperature**
A hyperparameter of the loss function. The default `0.5` is standard. A lower value (e.g. 0.1) makes the model more strict in its comparisons.

#### k-NN evaluation at checkpoints

If you enable this option, the interface will periodically evaluate the quality of the learned embeddings using a k-NN classifier (finding the k nearest images in embedding space and checking if they belong to the same class). This is useful to track whether the model is genuinely improving over time.

You must provide a labelled dataset with the standard subfolder structure. You can point to the same folder as your training dataset.

#### Augmentation parameters (expandable panel)

Augmentations are the random transformations applied to each image to create the multiple views. They are the engine of SimCLR: the model learns that two differently augmented versions of the same photo represent the same thing.

| Augmentation | Effect | Default |
|---|---|---|
| Color Jitter | Modifies brightness, contrast, saturation | Prob: 0.8, Strength: 0.5 |
| Horizontal Flip | Mirrors the image left-right | Prob: 0.5 |
| Vertical Flip | Flips the image upside down | Prob: 0.0 (disabled) |
| Grayscale | Converts to greyscale | Prob: 0.2 |
| Gaussian Blur | Applies a light blur | Prob: 0.5 |
| Random Rotation | Rotates the image by a random angle | Disabled by default |

> **Tip:** Click **Preview augmentations** before training to see exactly how your images will be transformed.

### Launching training

1. Fill in the dataset path
2. Choose a save path for the model
3. Adjust the Batch Size according to your hardware
4. Set a number of epochs (10-30 to start)
5. Click **Launch training**

### Monitoring progress

During training, you will see:

- **4 live metric tiles:** current epoch, loss, best loss so far, k-NN accuracy (if enabled)
- **A loss curve** plotted in real time — it should decrease progressively
- If k-NN is enabled, a **second orange curve** on the right axis shows the accuracy climbing

**What is a good loss?** The absolute value is not very meaningful — what matters is that it **decreases** over epochs. If it plateaus or goes back up, try a smaller learning rate.

### Files produced

```
runs/
+-- checkpoints/
|   +-- checkpoint_epoch_10.pth    (model state saved at epoch 10)
|   +-- checkpoint_epoch_20.pth
|   +-- ...
+-- best_model.pth                  (best model: minimum loss or best k-NN)
+-- simclr_model_last.pth           (model at the very last epoch)
+-- simclr_model_training_log.csv   (epoch-by-epoch metrics history)
```

Download buttons appear at the end of training so you can save files directly without navigating your filesystem.

---

## 6. Training a SupCon model

Go to **SupCon Training** in the sidebar.

### Difference from SimCLR

SupCon is designed for datasets with class labels. Instead of only saying "these two views come from the same image", SupCon says "all images from the same class should be close together, regardless of which specific photo they come from".

In practice, this produces better-structured embeddings where classes form tighter, more distinct clusters.

### When to choose SupCon over SimCLR?

- You have labels for your images -> SupCon
- You have no labels -> SimCLR
- You want to pre-train and then fine-tune -> SimCLR first, then supervised fine-tuning
- You want the best possible embeddings directly using your labels -> SupCon

### SupCon-specific parameters

Most parameters are identical to SimCLR. Here are the differences:

**SupCon temperature (tau)**
The recommended value `0.07` from the original SupCon paper is much lower than SimCLR's `0.5`, because SupCon works with supervised positives and can afford to be more discriminating.

**samples_per_image**
In memory-efficient mode, this controls how many positive pairs are sampled per image per batch. `2` is a good balance between quality and memory usage.

### Expected results

The SupCon loss is structurally different from the SimCLR loss (it accounts for class structure), so absolute values are not comparable between the two methods. What matters: the loss should decrease.

---

## 7. Generating embeddings

Go to **Generate Embeddings** in the sidebar.

### What is an embedding?

An embedding is the "visual signature" of an image produced by the trained model. It is a vector of 2048 numbers (for ResNet50) that encodes what the model has learned about that image.

If two images show the same type of object, their embeddings will be close in this 2048-dimensional space.

### How to generate embeddings

1. **Image folder:** point to the folder containing your images (subfolders are not required — all images in the folder and its subfolders will be processed)
2. **Upload a model (.pth):** upload your trained file (e.g. `best_model.pth`)
3. **CSV save path:** where to save the results
4. Click **Generate embeddings**

A progress bar shows the current image count as extraction proceeds.

### Format of the produced CSV

```
filename,feat_0,feat_1,...,feat_2047
cat_001.jpg,0.234,-0.112,0.891,...
cat_002.jpg,0.198,-0.089,0.923,...
dog_001.jpg,-0.445,0.334,-0.120,...
```

This file is used in the Visualization page to produce 2D plots.

---

## 8. Validating your model

Go to **Validation** in the sidebar.

### Principle

Validation trains a **linear classifier** on top of your model's embeddings and evaluates its accuracy. This is the standard way to assess a self-supervised model: if the embeddings are well-structured, even a simple classifier on top should perform well.

The backbone (ResNet50) is frozen, and only a small classification layer is trained on top of it. If accuracy is high, the model has learned useful visual representations.

### Parameters

- **Path to trained model:** your `best_model.pth` file
- **Validation dataset:** a labelled dataset (subfolder structure). The interface automatically splits it 80% train / 20% validation.
- **Batch Size:** adjust according to your available memory

### Metrics displayed

- **Accuracy:** percentage of correctly classified images
- **F1-Score:** a more robust metric than accuracy, especially when your classes are imbalanced
- **Cohen's Kappa:** measures prediction-vs-ground-truth agreement accounting for chance (0 = random, 1 = perfect)

### Interpreting results

| Accuracy | Interpretation |
|---|---|
| Below 50% | Barely above chance — review your dataset or increase epochs |
| 50-70% | Modest results — try more epochs or a larger backbone |
| 70-85% | Good results for contrastive learning |
| Above 85% | Excellent results |

---

## 9. GradCAM visualisation

Go to **GradCAM** in the sidebar.

### What is GradCAM?

GradCAM (Gradient-weighted Class Activation Mapping) produces a **heatmap** overlaid on the image, showing which regions of the image most activated the neural network. In plain terms: where is the model "looking" when it creates an embedding?

It is an interpretability tool. If your model is learning to recognise butterflies, the heatmap should be concentrated on the wings — not on the background.

### How to use it

**Single image mode:**
1. Select "Single Image"
2. Upload an image
3. Enter the path to your trained model
4. Click Run Grad-CAM

**Folder mode:**
1. Select "Folder"
2. Enter the path to a folder of images
3. Enter an output folder where the result images will be saved
4. Click Run Grad-CAM

### Reading heatmaps

- **Red / warm zones:** regions that most influenced the embedding
- **Blue / cool zones:** regions with little influence

**A good heatmap:** heat is concentrated on the object of interest (animal, plant, specimen).

**A poor heatmap:** heat is scattered over the background or irrelevant details. This means the model has not learned to focus on the object. Possible fixes: train for more epochs, use more data, or remove image backgrounds before training.

---

## 10. 2D embedding visualisation

Go to **Visualization** in the sidebar.

### Principle

Embeddings are 2048-dimensional vectors — impossible to visualise directly. We use **dimensionality reduction** to project them into 2D, while preserving proximity relationships as much as possible.

The interface offers three methods:

| Method | Advantages | Disadvantages |
|---|---|---|
| PCA | Very fast, deterministic | Poor at preserving non-linear structure |
| t-SNE | Excellent cluster visualisation | Slow, perplexity parameter to tune |
| UMAP | Fast, good global structure preservation | Requires the umap-learn package |

### Parameters

- **Path to embeddings CSV:** the file produced in the Generate Embeddings page
- **Label source:** either a CSV file with an `id` column (filename without extension) and a label column, or the dataset folder structure (the interface determines which subfolder each image belongs to)
- **Reduction method:** PCA, t-SNE or UMAP
- **k for Hit-Rate and k-NN:** used for automatic embedding quality evaluation
- **Minimum images per class:** classes below this threshold are excluded from k-NN evaluation (but still appear on the scatter plot)

### Reading the plot

Each dot represents one image. Dots of the same colour belong to the same class.

**A well-trained model produces a plot where:**
- Same-colour dots form tight clusters
- Different-colour clusters are clearly separated

**An under-trained model produces a plot where:**
- Colours are randomly mixed
- No visible structure exists

Hover over a dot to see the image filename. Click on a class in the legend to hide or show it. Zoom with the scroll wheel or the toolbar.

### Automatic evaluation metrics

**Hit-Rate@k:** for each image, among its k nearest neighbours, is there at least one from the same class? A value close to 1.0 indicates good embeddings.

**k-NN accuracy (5-fold CV):** accuracy of a k-NN classifier evaluated with 5-fold cross-validation. This is the most reliable way to quantify embedding quality without training a separate classifier.

---

## 11. Practical tips and troubleshooting

### Recommended workflow for a new project

```
1. Organise images into subfolders (one per class)
         |
         v
2. Run a quick test with 5-10 epochs to check everything works
         |
         v
3. Check GradCAM: is the model looking at the right thing?
         |
         v
4. Yes -> run a full training (50-200 epochs)
   No  -> review image preparation (remove background, crop tighter)
         |
         v
5. Generate embeddings over the full dataset
         |
         v
6. Visualise in 2D and inspect the clusters
         |
         v
7. Evaluate with Validation if you have class labels
```

### Common issues

**"CUDA out of memory"**
Reduce the Batch Size. Halve it until the error disappears.

**Loss does not decrease, or goes back up**
- Try a smaller learning rate (e.g. `0.0001`)
- Check that your images are not corrupted
- Increase the batch size if possible — very small batches produce an unstable loss

**The interface is very slow**
Without a GPU this is expected. Use 100-200 images for testing to check that everything works before committing to a full run.

**"No images found" error**
Your images must be inside subfolders within the folder you specified. A flat folder full of images without any subfolders will not work.

**The 2D plot shows no clusters**
- The model may not have trained long enough — try more epochs
- Your classes may be visually very similar
- Try t-SNE instead of PCA
- If you have labels, try SupCon instead of SimCLR

**On Windows: DataLoader crashes on startup**
Expected behaviour. The interface automatically detects Windows and sets `num_workers=0` to prevent multiprocessing errors.


### Saving and reproducibility

The interface automatically saves:
- A checkpoint every N epochs (configurable in the UI)
- The best model, based on loss or k-NN accuracy
- The last model at the end of training
- A CSV log with all epoch-by-epoch metrics

### Understanding the save format

Every `.pth` file produced by `SimCLR_GUI4.py` contains not only the model weights but also its architecture metadata:

```python
{
    "model_state": ...,          # the learned weights
    "backbone":    "resnet50",   # which backbone was used
    "out_dim":     128,          # projection head size
    "saved_at":    "2024-11-03T14:32:11"
}
```

Checkpoints additionally contain:

```python
{
    "model_state":     ...,
    "optimizer_state": ...,
    "epoch":           42,
    "hparams":         { ... },  # augmentation parameters used during training
    "saved_at":        "2024-11-03T14:32:11"
}
```

This means that when you load a model in the Generate Embeddings or Validation pages, the interface automatically reconstructs the correct architecture — no need to remember whether you used resnet18 or resnet50.
