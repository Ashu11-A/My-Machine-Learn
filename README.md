<div align="center">

# My Machine Learn

![License](https://img.shields.io/github/license/Ashu11-A/IDP-Machine-Learn?style=for-the-badge&color=302D41&labelColor=f9e2af&logoColor=302D41)
![Stars](https://img.shields.io/github/stars/Ashu11-A/IDP-Machine-Learn?style=for-the-badge&color=302D41&labelColor=f9e2af&logoColor=302D41)
![Last Commit](https://img.shields.io/github/last-commit/Ashu11-A/IDP-Machine-Learn?style=for-the-badge&color=302D41&labelColor=b4befe&logoColor=302D41)
![Repo Size](https://img.shields.io/github/repo-size/Ashu11-A/IDP-Machine-Learn?style=for-the-badge&color=302D41&labelColor=90dceb&logoColor=302D41)

<br>

<p align="center">
  <strong>Machine Learning Pipeline with GPU acceleration (NVIDIA CUDA) for diabetes classification, developed by <a href="https://github.com/Ashu11-A">@Ashu11-A</a>.</strong>
  <br><br>
  <sub>
    Training of multiple classifiers with <strong>Early Stopping</strong>,
    interactive 3D visualization rendered on the GPU via <strong>OpenGL (vispy)</strong>,
    and real-time camera synchronization across all plots.
  </sub>
</p>

<p align="center">
  <a href="https://github.com/Ashu11-A/IDP-Machine-Learn/stargazers">
    <img src="https://img.shields.io/badge/Leave%20a%20Star%20🌟-302D41?style=for-the-badge&color=302D41&labelColor=302D41" alt="Star Repo">
  </a>
</p>

</div>

---

## 📋 Table of Contents

- [About the Project](#-🧠-about-the-project)
- [Dataset](#-dataset)
- [Architecture](#-architecture)
- [Models Used](#-models-used)
- [Early Stopping](#-early-stopping)
- [Training Results](#-training-results)
- [3D GPU Visualization](#-3d-gpu-visualization)
- [Requirements](#-requirements)
- [Installation and Execution](#-installation-and-execution)
- [Project Structure](#-project-structure)

---

## 🧠 About the Project

This project implements a complete Machine Learning pipeline for **diabetes classification**, focusing on:

- **Full GPU Acceleration** — preprocessing, training, and rendering are executed in the video card's VRAM via NVIDIA CUDA and OpenGL.
- **Simultaneous model comparison** — three classification algorithms are trained and compared side-by-side.
- **Automated hyperparameter search** — Early Stopping with minimum improvement tolerance (`min_delta`) prevents search overfitting and premature stops due to noise.
- **Interactive 3D visualization** — plots display decision boundaries in 3D space (Insulin × Glucose × BMI), with a synchronized camera across all panels.

---

## 📊 Dataset

| Field        | Detail |
|:-------------|:-------|
| **Source** | [Kaggle — Diabetes Dataset (John Da Silva)](https://www.kaggle.com/datasets/johndasilva/diabetes) |
| **Samples** | 2,000 patients |
| **Features used** | `Insulin`, `Glucose`, `BMI` |
| **Target** | `Outcome` — `0` (non-diabetic) · `1` (diabetic) |
| **Split** | 70% train · 30% test (no shuffling, temporal order preserved) |
| **Scaling** | MinMaxScaler → `[-1, 1]` range in `float32` |

---

## 🏗️ Architecture

The project follows a layered architecture with a clear separation of responsibilities. Each layer is an independent Python subpackage:


```

IDP-Machine-Learn/
│
├── main.py                  ← single entry point
│
└── diabetes_ml/             ← project namespace
├── config.py                ← PipelineConfig (frozen dataclass)
├── pipeline.py              ← DiabetesMLPipeline (orchestrator)
│
├── data/
│   ├── dataset.py           ← ProcessedDataset (CPU + GPU arrays)
│   └── pipeline.py          ← DataPipeline (load → split → scale)
│
├── training/
│   ├── early_stopping.py    ← EarlyStopping + EarlyStoppingState
│   ├── wrappers.py          ← Strategy: GPUModelWrapper + 3 models
│   └── tuner.py             ← HyperparameterTuner (search loop)
│
└── visualization/
├── gpu_canvas.py            ← GPUScatterRow (vispy OpenGL)
├── grid.py                  ← DecisionBoundaryGrid (3D mesh on GPU)
├── subplots.py              ← ModelSubplotBuilder (Markers via OpenGL)
├── tuning_plot.py           ← TuningPlotBuilder (matplotlib)
└── interaction.py           ← DiabetesMLWindow (Qt window)

```

**Applied design patterns:**

| Pattern | Where |
|:--------|:------|
| **Strategy** | `GPUModelWrapper` — adding a new model requires only a new subclass |
| **Dataclass (frozen)** | `PipelineConfig` — immutable and hashable configuration |
| **Dependency Injection** | Shared camera injected into both `GPUScatterRow` components |
| **Single Responsibility** | Each file contains exactly one responsibility |

---

## 🤖 Models Used

### K-Nearest Neighbors (KNN) — via cuML

KNN classifies a point based on the **K nearest neighbors** in the feature space. For each new sample, the algorithm calculates the Euclidean distance to all training points and assigns the majority class among the closest K.

- **Searched hyperparameter:** `K` (number of neighbors) — values from 20 to 1,024
- **Library:** `cuml.neighbors.KNeighborsClassifier` (100% GPU execution)
- **Pros:** simple, no explicit training phase, interpretable
- **Cons:** slow inference for large datasets; sensitive to features on different scales (hence MinMaxScaler is essential)


```

Best K found: 20   →   Test accuracy: 76.50%

```

---

### Random Forest (RF) — via cuML

Random Forest is an **ensemble of decision trees** trained on random subsets of the data (bagging) and with random subsets of features at each split. The final prediction is made by majority voting among all trees.

- **Searched hyperparameter:** `n_estimators` (number of trees)
- **Fixed configuration:** `max_depth=5`, `random_state=42`
- **Library:** `cuml.ensemble.RandomForestClassifier` (parallel training on GPU)
- **Pros:** robust to overfitting, performs well without extensive tuning, naturally parallel
- **Cons:** less interpretable than a single tree; can be slow with many deep trees


```

Best N found: 20   →   Test accuracy: 79.00%

```

---

### Gradient Boosting (GB) — via XGBoost + CUDA

Gradient Boosting builds trees **sequentially**: each new tree is trained to correct the residual errors of the previous tree, minimizing a loss function via gradient descent.

- **Searched hyperparameter:** `n_estimators` (number of estimators/rounds)
- **Fixed configuration:** `max_depth=3`, `tree_method='hist'`, `device='cuda'`, `random_state=42`
- **Library:** `xgboost.XGBClassifier` with native CUDA backend
- **Pros:** generally the highest accuracy model among the three; efficient with `tree_method='hist'`; natively accepts CuPy arrays without CPU↔GPU transfer overhead
- **Cons:** more sensitive to hyperparameters; sequential training limits parallelism compared to RF


```

Best N found: 20   →   Test accuracy: 80.50%

```

---

## ⏱️ Early Stopping

The hyperparameter search uses **Early Stopping with a minimum improvement tolerance**, avoiding two common issues:

1. **Premature stopping due to noise** — small negative oscillations do not interrupt the search
2. **Unnecessarily long search** — if no model improves significantly for `patience_limit` consecutive steps, the search ends

```python
# diabetes_ml/config.py
patience_limit: int = 150    # steps without improvement before stopping
min_delta: float     = 0.01  # minimum improvement considered significant
initial_param: int   = 20    # initial value of the searched hyperparameter

```

**Decision logic at each step:**

```
new_accuracy - best_accuracy > min_delta?
    ├── YES → updates best, resets patience
    └── NO  → increments patience
               └── patience >= patience_limit → stops for this model

```

The three models are searched **in parallel** step by step. The global search ends when **all** models hit their patience limit.

---

## 📈 Training Results

Log of the last step before stopping:

```
Step 170 | KNN: 0.7600 (Best: 0.7650 | Pat: 150/150)
         |  RF: 0.7950 (Best: 0.7900 | Pat: 150/150)
         |  GB: 0.8950 (Best: 0.8050 | Pat: 150/150)
Search completed via Early Stopping!

```

| Model | Best Parameter | Best Accuracy (Test) |
| --- | --- | --- |
| KNN | K = 20 | **76.50%** |
| Random Forest | N = 20 | **79.00%** |
| Gradient Boosting | N = 20 | **80.50%** |

**Gradient Boosting via XGBoost** achieved the highest accuracy, which is expected given that boosting algorithms tend to outperform bagging methods and simple instances when the data has complex non-linear relationships between features.

---

## 🎮 3D GPU Visualization

The visualization was reimplemented from **matplotlib 3D (CPU)** to **vispy OpenGL (GPU)**:

| Aspect | Matplotlib (before) | vispy OpenGL (now) |
| --- | --- | --- |
| Rendering | Software (CPU) | OpenGL (GPU/VRAM) |
| Framerate when rotating | Low (~2–5 fps) | High (60+ fps) |
| Data buffer | Recalculated every frame | Sent once to VRAM |
| Synchronization | Event callbacks | Shared camera object |

**Camera synchronization:** all 6 ViewBoxes (3 models × 2 splits) share the **same** `TurntableCamera` instance. Rotating any plot instantly rotates all of them — with no event overhead, by Python reference.

```python
shared_camera = TurntableCamera(fov=40, elevation=25, azimuth=45)
train_row = GPUScatterRow(n_models, shared_camera)   # ← same camera
test_row  = GPUScatterRow(n_models, shared_camera)   # ← same camera

```

**Color legend:**

| Color | Meaning |
| --- | --- |
| 🔵 Blue | Class 0 — correct prediction |
| 🔴 Red | Class 1 — correct prediction |
| 🟢 Mint Green | Class 0 — incorrect prediction |
| 🟡 Yellow | Class 1 — incorrect prediction |

---

## 📦 Requirements

* Python `>= 3.11`
* NVIDIA GPU with CUDA support `>= 11.8`
* CUDA Toolkit installed on the system

**Main dependencies:**

| Package | Usage |
| --- | --- |
| `cuml` | GPU-accelerated KNN and Random Forest |
| `xgboost` | Gradient Boosting with CUDA backend |
| `cupy` | Arrays in VRAM and GPU↔CPU transfers |
| `vispy` | 3D Rendering via OpenGL |
| `PyQt5` / `PyQt6` | Window backend for vispy + matplotlib |
| `matplotlib` | Fine Tuning Plot (accuracy vs parameter) |
| `scikit-learn` | `train_test_split`, `MinMaxScaler` |
| `pandas` / `numpy` | Data manipulation |

---

## 🚀 Installation and Execution

```bash
# 1. Clone the repository
git clone [https://github.com/Ashu11-A/IDP-Machine-Learn.git](https://github.com/Ashu11-A/IDP-Machine-Learn.git)
cd IDP-Machine-Learn

# 2. Place the dataset in the project root
#    Download at: https://www.kaggle.com/datasets/johndasilva/diabetes

# 3. Install dependencies (recommended: uv)
uv sync

# 4. Run
uv run main.py
# or:
python main.py

```

> **Note:** `cuml` requires [RAPIDS](https://rapids.ai/start/) installed. The simplest way is via `conda`:
> ```bash
> conda install -c rapidsai -c conda-forge cuml cuda-version=12.0
> 
> ```
> 
> 

To customize the search parameters without modifying the code:

```python
from diabetes_ml.config import PipelineConfig
from diabetes_ml.pipeline import DiabetesMLPipeline

cfg = PipelineConfig(
    patience_limit=200,
    min_delta=0.005,
    initial_param=10,
)
DiabetesMLPipeline(cfg).run()

```

---

## 📁 Project Structure

```
IDP-Machine-Learn/
├── main.py
├── diabetes.csv               ← dataset (not included, download from Kaggle)
├── STRUCTURE.md
├── pyproject.toml
├── uv.lock
└── diabetes_ml/
    ├── __init__.py
    ├── config.py
    ├── pipeline.py
    ├── data/
    │   ├── __init__.py
    │   ├── dataset.py
    │   └── pipeline.py
    ├── training/
    │   ├── __init__.py
    │   ├── early_stopping.py
    │   ├── wrappers.py
    │   └── tuner.py
    └── visualization/
        ├── __init__.py
        ├── gpu_canvas.py
        ├── grid.py
        ├── subplots.py
        ├── tuning_plot.py
        └── interaction.py

```

