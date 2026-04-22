# 🔬 tuto_simclr

**Train visual representation models (SimCLR & SupCon) using a simple graphical interface**

---

## 🧠 What is SimCLR / SupCon?

SimCLR and SupCon are **visual representation learning** methods. Instead of training a model to recognise predefined categories like a standard classifier, they train a model to produce **visual fingerprints** (embeddings) such that similar images have similar fingerprints, and different images have distant ones.

In practice, this allows you to:
- Cluster visually similar images without having labelled them (SimCLR)
- Strengthen those clusters using class labels (SupCon)
- Visualise the structure of your dataset in 2D
- Retrieve images by visual similarity

---

## 📁 Repository structure

```
tuto_simclr/
├── SimCLR_GUI4.py          # Main Streamlit interface
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── TUTORIAL.md             # Step-by-step tutorial
└── data/
    ├── class_A/            # One subfolder = one class
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class_B/
    │   ├── image1.jpg
    │   └── ...
    └── class_C/
        └── ...
```

> **Important:** the `data/` folder contains example images organised into subfolders. Each subfolder corresponds to one category. You can replace these images with your own, keeping the same structure.

---

## 🚀 Quick start

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) (or [Miniconda](https://docs.anaconda.com/miniconda/)) installed on your machine — this handles Python and package management for you
- [Git](https://git-scm.com/downloads/) installed
- (Optional but recommended) an NVIDIA GPU with [CUDA drivers](https://developer.nvidia.com/cuda-downloads)

> **Why Anaconda?** Anaconda makes it easy to create isolated Python environments and avoids common conflicts between packages. It also ships with many scientific libraries pre-installed.

---

### Step 1 — Install Anaconda

Download and install **Anaconda** from [anaconda.com/download](https://www.anaconda.com/download).  
During installation, accept the default options. On Windows, you can use the **Anaconda Prompt** that gets installed alongside it — use that instead of the regular Command Prompt for all the steps below.

To verify it is installed correctly, open **Anaconda Prompt** (Windows) or a terminal (Mac/Linux) and type:

```bash
conda --version
# You should see something like: conda 24.x.x
```

---

### Step 2 — Clone the repository

```bash
git clone https://github.com/your_username/tuto_simclr.git
cd tuto_simclr
```

If you do not have Git installed, click **Code > Download ZIP** on the GitHub page, extract the archive, and navigate into the folder.

---

### Step 3 — Create a conda environment

A conda environment is an isolated workspace with its own Python version and packages. This prevents conflicts with other projects on your machine.

```bash
conda create -n simclr_env python=3.10 -y
conda activate simclr_env
```

After activation, you should see `(simclr_env)` at the start of your prompt. **All subsequent commands must be run with this environment active.**

---

### Step 4 — Install PyTorch

PyTorch needs to be installed separately so you can pick the right version for your hardware.

**If you have an NVIDIA GPU** (recommended for training):

Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), select your OS and CUDA version, and copy the generated install command. It will look something like:

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**If you have no GPU (CPU only):**

```bash
conda install pytorch torchvision cpuonly -c pytorch -y
```

> **How to check your CUDA version:** open a terminal and run `nvidia-smi`. The CUDA version appears in the top-right corner of the output. If the command is not found, you either have no NVIDIA GPU or the drivers are not installed.

---

### Step 5 — Install the remaining dependencies

```bash
pip install -r requirements.txt
```

---

### Step 6 — Launch the interface

```bash
streamlit run SimCLR_GUI4.py
```

The interface opens automatically in your browser at `http://localhost:8501`.  
To stop it, go back to the terminal and press `Ctrl + C`.

---

### Reopening the interface later

Every time you want to use the interface after closing it, you just need to:

```bash
# 1. Open Anaconda Prompt (Windows) or a terminal (Mac/Linux)
# 2. Activate the environment
conda activate simclr_env

# 3. Navigate to the project folder
cd path/to/tuto_simclr

# 4. Launch the interface
streamlit run SimCLR_GUI4.py
```

---

## 🖥️ Interface overview

The sidebar lets you navigate between six pages:

| Page | Purpose |
|---|---|
| **Training** | Train a SimCLR model (self-supervised) |
| **SupCon Training** | Train a SupCon model (supervised contrastive) |
| **Generate Embeddings** | Extract visual fingerprints from a folder of images |
| **Validation** | Evaluate a trained model with a linear classifier |
| **GradCAM** | Visualise where the model "looks" in an image |
| **Visualization** | Plot embeddings in 2D and evaluate k-NN metrics |

---

## 📁 Data folder

The `data/` folder included in this repository contains example images already organised into subfolders, so you can test the interface immediately. Simply point any path field in the interface to `data/` to get started.

To use your own images, create a folder with the following structure — one subfolder per class, named after the class:

```
my_dataset/
├── cats/
│   ├── cat_001.jpg
│   └── cat_002.jpg
├── dogs/
│   └── dog_001.jpg
└── birds/
    └── bird_001.jpg
```

Accepted image formats: `.jpg`, `.jpeg`, `.png`

---

## ⚙️ requirements.txt

The following packages are installed via `pip install -r requirements.txt` (after PyTorch has been installed via conda in Step 4):

```
streamlit
pandas
matplotlib
pillow
tqdm
scikit-learn
typing
numpy
opencv-python
plotly
pathlib
argparse
grad-cam
umap-learn
scipy
distinctipy
```

> **Note:** `torch` and `torchvision` are intentionally absent from this file — they must be installed via conda as described in Step 4, to ensure you get the version that matches your hardware (GPU or CPU).

---

## 💡 Frequently asked questions

**I get `CUDA out of memory`.**
Reduce the Batch Size in the interface (try 16 or 8).

**Training is very slow.**
Without a GPU, training on CPU can take hours. Start with 5–10 epochs just to verify everything works.

**My images are not found.**
Make sure your images are inside **subfolders** within the folder you specified. A flat folder of images with no subfolders will not work.

**I want to use my own images.**
Organise them into subfolders named after each category (as in `data/`), then enter the path to your folder in the interface.

**The interface crashes on Windows with DataLoader errors.**
This is a known Windows multiprocessing issue with PyTorch. The interface handles this automatically by using `num_workers=0` on Windows — no action needed on your part.

**How do I stop the application?**
Go back to the terminal and press `Ctrl + C`.

---

## 📖 Full tutorial

A complete step-by-step tutorial (concepts + interface walkthrough) is available in **`TUTORIAL.md`**.

It covers:
1. Preparing your dataset
2. Training with SimCLR
3. Training with SupCon
4. Generating embeddings
5. Validating your model
6. GradCAM visualisation
7. 2D embedding visualisation

---

## 📄 Licence

MIT — free to use, modify, and distribute.
