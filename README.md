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

- [Python 3.9 or 3.10](https://www.python.org/downloads/) installed on your machine
- [Git](https://git-scm.com/downloads/) installed
- (Optional but recommended) an NVIDIA GPU with [CUDA drivers](https://developer.nvidia.com/cuda-downloads)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your_username/tuto_simclr.git
cd tuto_simclr

# 2. Create a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the interface
streamlit run SimCLR_GUI4.py
```

The interface opens automatically in your browser at `http://localhost:8501`.

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

```
streamlit>=1.32
torch>=2.0
torchvision>=0.15
Pillow>=9.0
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
scikit-learn>=1.3
tqdm>=4.65
pytorch-grad-cam>=1.4
scipy>=1.11
umap-learn>=0.5
distinctipy>=1.2
plotly>=5.18
```

> **GPU setup:** if you have an NVIDIA GPU, replace the `torch` and `torchvision` lines with the CUDA-enabled versions from [pytorch.org](https://pytorch.org/get-started/locally/).

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
This is a known Windows multiprocessing issue with PyTorch. Set `num_workers=0` in any DataLoader call if you encounter it.

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
