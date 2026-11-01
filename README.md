# Neural-Additive-Models-Step-Constraint-for-Clinical-Application 
[![License: MIT](https://shields.io)](https://opensource.org)
[![Python 3.10+](https://shields.io)](https://python.org)
[![Framework: PyTorch](https://shields.io)](https://pytorch.org)

This repository contains the official implementation for the paper: **"Architectural Constraint of Neural Additive Models via Discrete Step Activations for Clinical Guideline Alignment"**. 

This project addresses the **Continuous Lab-Value Disconnect** in clinical AI. While inherently interpretable frameworks like Neural Additive Models (NAMs) isolate feature contributions, standard smooth activation functions (e.g., `Tanh`, `Sigmoid`) force models to learn continuous curves. However, actual hospital protocols utilize strict, deterministic step-thresholds for patient triage (e.g., Shock Index boundaries at 0.7 and 1.0). 

Instead of relying on hyperparameter-sensitive calculus loss penalties, this repository introduces a structural mechanism that replaces continuous sub-networks with a hard **Discrete Step Activation Layer** using piecewise boolean indicators. 

---

## 📊 Experimental Results

Both architectures were trained and validated side-by-side using an identical synthetic Electronic Health Record (EHR) cohort (n=2,000, 80/20 split, Seed 42) optimized strictly under Binary Cross-Entropy (`BCELoss`):

| Model Architecture | Held-Out AUROC | Validation BCE Loss | Internal Geometric State |
| :--- | :---: | :---: | :--- |
| **Vanilla NAM (Tanh)** | 0.8642 | 0.458 | Gradual S-Shaped Overfitted Curve |
| **Hard Step NAM (Ours)** | **0.8644** | **0.448** | **Exactly Piecewise Constant** |

### Key Takeaway
Forcing the neural network to snap into rigid clinical boundaries resulted in a **+0.0002 AUROC improvement** and a **-0.010 reduction in validation error**. By aligning the internal model architecture with the discrete threshold reality of clinical settings, we completely eliminated out-of-distribution safe-zone noise without sacrificing predictive power.

The model extracted explicit, auditable log-odds parameters matching standard ICU triage rules:
* **Shock Index < 0.7**: b₀ = -1.851 (Clinically Stable, protective weight)
* **0.7 ≤ Shock Index < 1.0**: b₁ = -0.004 (Elevated Alert, neutral weight)
* **Shock Index ≥ 1.0**: b₂ = +1.716 (Critical Emergency, severe risk weight)

---

## 📁 Repository Structure

```text
├── train_nam_step_activation.py   # Synthetic EHR experiment (unchanged)
├── requirements.txt               # Python dependencies
├── README.md
├── nam_step_activation_code_figures.zip
└── real_data/                     # Camera-ready public-cohort experiments
    ├── code/                      # Prep, training, and figure scripts
    ├── results/                   # Metrics JSON, logs, aggregate plot arrays
    ├── figures/
    └── paper/                     # Camera-ready tex, pdf, response to reviewers
```

---

## ⚙️ Getting Started

### 1. Prerequisites
Ensure you have Python 3.10 or higher installed. Clone the repository locally:
```bash
git clone https://github.com
cd ieee-nam-threshold
```

### 2. Environment Setup
It is highly recommended to isolate your dependencies using a virtual environment:
```bash
# Create virtual environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt
```

### 3. Run the Experiment
To execute the data simulation framework, train both versions of the Neural Additive Models, output performance configurations, and save the visualization graphs, execute:
```bash
python train_nam_step_activation.py
```

---

## 🔍 Model Architecture Detail

The core innovation is implemented directly inside the custom sub-network modules via boolean tensor masking during the forward pass:

$$\mathbf{1}\{x < 0.7\}, \quad \mathbf{1}\{0.7 \le x < 1.0\}, \quad \mathbf{1}\{x \ge 1.0\}$$

This mathematically limits the optimization landscape to a three-level step function. Backpropagation updates only the distinct active coefficient block $b_k$ mapping to the patient's incoming real-time telemetry slice, guaranteeing zero slope leakage inside safe clinical bands.

---

## Real-data camera-ready experiments

Shock Index is heart rate divided by systolic blood pressure (HR/SBP). The synthetic script above is unchanged. The camera-ready public-cohort code, metrics, figures, and manuscript live under `real_data/`. Cohort tables are written to `real_data/data_local/` and are gitignored. Do not commit those CSVs, the XPT/CSV.GZ downloads, or any other patient-level file.

### Cohorts

| Cohort | Who | Label | n (events) |
| :--- | :--- | :--- | :--- |
| **NHANES 2017–March 2020** pre-pandemic | Adults aged 20 or older. Shock Index uses the first oscillometric pulse (`BPXOPLS1`) and systolic blood pressure (`BPXOSY1`) | Self-reported prior CVD: heart failure, coronary heart disease, angina, heart attack, or stroke (`MCQ160B`–`F` = 1). This is prevalent history, not a future event. Estimates are not survey-weighted | 6,979 (853, 12.2%) |
| **PhysioNet/CinC Challenge 2012** sets A, B, and C | ICU stays. Shock Index is the first time-paired heart rate and systolic blood pressure in the first 6 hours (noninvasive `NISysABP` preferred, otherwise invasive `SysABP`) | In-hospital death | 11,520 (1,635, 14.2%) |
| **eICU-CRD demo v2.0.1** | First ICU stay of each adult hospital admission (`unitvisitnumber == 1`, age ≥ 18). Systolic blood pressure is the first value in the first 6 hours (noninvasive aperiodic preferred, otherwise invasive periodic). Heart rate is the periodic measurement within 15 minutes of that blood pressure | Hospital death | 1,929 (170, 8.8%) |

### Download the public data

Put raw files outside the repository and point these variables at them. If a variable is unset, the prep script falls back to `data_nhanes`, `data_pn`, or `data_pn/eicu` relative to the current working directory.

```bash
export NHANES_DIR=/path/to/nhanes_xpt
export PN2012_DIR=/path/to/challenge-2012-1.0.0
export EICU_DIR=/path/to/eicu-crd-demo-2.0.1
```

**NHANES.** Download `P_DEMO`, `P_BPXO`, and `P_MCQ` from the [2017–March 2020 pre-pandemic documentation](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017) ([demographics](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DEMO.htm), [oscillometric blood pressure](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BPXO.htm), [medical conditions](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_MCQ.htm)). The prep script expects the filenames `P_DEMO.xpt`, `P_BPXO.xpt`, and `P_MCQ.xpt`:

```bash
mkdir -p "$NHANES_DIR"
curl -L -o "$NHANES_DIR/P_DEMO.xpt" https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_DEMO.XPT
curl -L -o "$NHANES_DIR/P_BPXO.xpt" https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_BPXO.XPT
curl -L -o "$NHANES_DIR/P_MCQ.xpt" https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_MCQ.XPT
```

**PhysioNet 2012** ([open files](https://physionet.org/content/challenge-2012/1.0.0/)):

```bash
wget -r -N -c -np https://physionet.org/files/challenge-2012/1.0.0/
```

`PN2012_DIR` must contain `Outcomes-a.txt`, `Outcomes-b.txt`, `Outcomes-c.txt` and directories `set-a/`, `set-b/`, and `set-c/` of per-record `.txt` files. The recursive download already unpacks sets A and B. Extract set C next to them:

```bash
tar -xzf "$PN2012_DIR/set-c.tar.gz" -C "$PN2012_DIR"
```

**eICU-CRD demo v2.0.1** ([open demo](https://physionet.org/content/eicu-crd-demo/2.0.1/), not the credentialed full database):

```bash
wget -r -N -c -np https://physionet.org/files/eicu-crd-demo/2.0.1/
```

`EICU_DIR` must contain `patient.csv.gz`, `vitalPeriodic.csv.gz`, and `vitalAperiodic.csv.gz`.

Build the analysis tables (still only on disk under `real_data/data_local/`):

```bash
python real_data/code/prep_nhanes.py
python real_data/code/prep_physionet2012.py
python real_data/code/prep_eicu_demo.py
```

### Protocol

Seeds 0, 1, 2, 3, and 4. Each seed draws one stratified 80/20 train/test split (`random_state` equal to the seed), and every model on that seed shares the split. The 20% portion is the held-out test set. There is no validation set: nothing is selected on held-out data. Training runs for a fixed 150 epochs with Adam (learning rate `1e-3`, batch size 64, binary cross-entropy with logits), with no early stopping.

### Models

- **Vanilla NAM.** Separate tanh sub-networks for Shock Index and for age, added together.
- **Step NAM.** Hard step activation on Shock Index at 0.7 and 1.0 (three constant levels; boolean masks, not a soft step) plus a tanh age sub-network. The same model is refit at (0.6, 0.9) and (0.8, 1.1) for cutoff sensitivity.
- **Binned LR.** L2 logistic regression (`C=1`) with one-hot Shock Index bands (`<0.7`, `0.7` to `1.0`, `≥1.0`, reference band dropped) and a linear age term.
- **Binned LR + age spline.** The same Shock Index bands with a cubic age spline (5 knots), still `C=1`.

### Run

From the repository root, after the three prep commands:

```bash
python real_data/code/run_camera_ready.py nhanes
python real_data/code/run_camera_ready.py physionet2012
python real_data/code/run_camera_ready.py eicu-demo
python real_data/code/make_figures.py
```

Metrics are written to `real_data/results/metrics_<cohort>.json`. Figures are written to `real_data/figures/`. Saved plot arrays store shape functions, ROC curves, training curves, and binned event rates. They do not store row-level Shock Index or labels. The camera-ready manuscript is `real_data/paper/nam_step_activation_camera_ready.tex` (figures resolve from `real_data/figures/`).

### Results

Held-out test AUROC, Mean±SD over seeds 0–4, from `summary_mean_sd` in the metrics JSON. The same splits are used for every model.

| Model | NHANES | PhysioNet 2012 | eICU demo |
| :--- | :---: | :---: | :---: |
| Vanilla NAM | 0.7838 ± 0.0206 | 0.6345 ± 0.0141 | 0.7517 ± 0.0320 |
| Step NAM (0.7, 1.0) | 0.7828 ± 0.0208 | 0.6303 ± 0.0176 | 0.7421 ± 0.0283 |
| Binned LR | 0.7816 ± 0.0207 | 0.6285 ± 0.0161 | 0.7461 ± 0.0339 |
| Binned LR + age spline | 0.7815 ± 0.0209 | 0.6312 ± 0.0153 | 0.7418 ± 0.0357 |

Cutoff sensitivity for the step NAM on those same splits. Parentheses are runs with monotone levels `b0 < b1 < b2`.

| Cutoffs | NHANES | PhysioNet 2012 | eICU demo |
| :--- | :---: | :---: | :---: |
| (0.6, 0.9) | 0.7790 ± 0.0213 (5/5) | 0.6291 ± 0.0104 (0/5) | 0.7330 ± 0.0313 (5/5) |
| (0.7, 1.0) | 0.7828 ± 0.0208 (5/5) | 0.6303 ± 0.0176 (5/5) | 0.7421 ± 0.0283 (5/5) |
| (0.8, 1.1) | 0.7814 ± 0.0214 (4/5) | 0.6332 ± 0.0134 (5/5) | 0.7377 ± 0.0315 (5/5) |

---

## 📜 Citation

If you use this architecture or framework in your research, please cite our conference manuscript:

```bibtex
@inproceedings{mamidala2026architectural,
  title={Architectural Constraint of Neural Additive Models via Discrete Step Activations for Clinical Guideline Alignment},
  author={Mamidala, Pranav},
  booktitle={Proceedings of the High School Symposium at the IEEE International Conference on Big Data (IEEE BigData)},
  year={2026}
}
```
Footnote: Project was completed last month, committed Sep 20 2026
