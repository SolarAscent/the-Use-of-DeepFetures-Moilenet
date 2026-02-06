# Feature Depth & Architecture Comparison: MobileNet-SVM vs. Vision Transformer

Use MobileNetV2 as a **fixed feature extractor** and compare how **shallow vs. deep** features affect **linear separability** on a **3-class** image dataset (**bird / cat / dog**).  

## Goal
This project investigates two fundamental questions in computer vision classification:

1.  **The Impact of Feature Depth on Linear Separability (CNN)**:
    * How does the "depth" of a feature representation affect its ability to be classified by a simple Linear SVM?
    * We treat **MobileNetV2** as a fixed feature extractor and probe it at **19 different depths** (from shallow edges to deep semantic features) to observe the evolution of accuracy.

2.  **CNN vs. Transformer Paradigm**:
    * How does a state-of-the-art **Vision Transformer (ViT-B/16)** compare against the best-performing "Deep CNN Feature + SVM" combination?

---

## Design

### Experiment A: Layer-wise Analysis (MobileNetV2 + Linear SVM)
* **Methodology**:
    1.  Freeze MobileNetV2 (ImageNet weights).
    2.  Extract feature maps from **every valid cut point** (Layers 1 to 19).
    3.  Flatten features and train a **Linear Support Vector Machine (LinearSVC)** for each layer depth.
* **Hypothesis**: Deeper layers should provide better semantic separation, but may hit a diminishing return or overfitting point.
* **Metrics**: Accuracy vs. Layer Depth curve.

### Experiment B: End-to-End Learning (Vision Transformer)
* **Methodology**:
    1.  Fine-tune a pre-trained **ViT-B/16** model.
    2.  Train with mixed-precision (AMP) for 5 epochs.
* **Goal**: To establish a high-performance baseline representing modern "Deep Learning" capabilities.
---

## Project Structure

```text
.
├── main_code.py           # [Exp A Runner] Runs MobileNet layer-wise extraction -> SVM -> Plotting
├── deep_main.py           # [Exp B Runner] Runs ViT training (End-to-End)
├── config.py              # Shared configuration (Batch size=128/256, Class names, Paths)
├── model_utils.py         # MobileNet utilities (Feature extraction logic & SVM classifier)
├── deep_model_utils.py    # ViT utilities (Optimized DataLoaders & Training loops)
├── visualization.py       # Smart Plotting (Handles both "Layer Depth" and "Epochs" x-axis)
├── data_utils.py          # Offline Data Augmentation & Splitting
└── requirements.txt       # Dependencies
```

---

## Dataset Layout

Place your raw images in **`data/`** (can be nested; the code uses recursive search):

In our training, we use this dataset:
https://www.kaggle.com/datasets/mahmoudnoor/high-resolution-catdogbird-image-dataset-13000/data


```
data/
  bird/  (can contain bird/*.jpg or bird/bird/*.jpg etc.)
  cat/
  dog/
```

The script will create `data_split/` automatically:

```
data_split/
  train/bird|cat|dog
  val/bird|cat|dog
  test/bird|cat|dog
```

Split ratio is **60% / 20% / 20%** (train/val/test) and is reproducible via `RANDOM_SEED`.

---

## Installation

### 1) Create environment (recommended)

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

> For PyTorch installation (CPU/CUDA), follow the official selector if needed.

---

1. Environment Setup (Recommended)

```Bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.\.venv\Scripts\Activate.ps1
```

## How to Run
### Experiment A: MobileNetV2 (Layer-wise Analysis)
This script will extract features from all 19 layers of MobileNetV2 and train separate SVMs for each.

```bash
python main_code.py
```
Output: result.png (Accuracy vs. Layer Depth curve)

Experiment B: Vision Transformer (ViT-B/16)
This script fine-tunes a ViT model.

Configuration: Check config.py to adjust BATCH size (Recommended: 128 or 256 for 24G+ VRAM).

```Bash
python deep_main.py
```

Output: result_vit_manual.png or similar (Accuracy vs. Epoch curve)

## Notes

- **Three-class** classification is enforced by the folder names in `data/` and `config.CLASS_NAMES`.
- The SVM is **LinearSVC** (one-vs-rest) so that multiclass hinge loss is well-defined from `decision_function`.
- Augmentation is **offline** (writes augmented images to disk in train/val folders). If you re-run augmentation repeatedly, it will skip already-augmented files based on `_aug` in the filename stem.

---

## License


