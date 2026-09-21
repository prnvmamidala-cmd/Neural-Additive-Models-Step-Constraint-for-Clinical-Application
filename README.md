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
├── train_nam_threshold.py      # Core runnable training script and pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation
└── clinical_step_results.png   # Visualization plot output (Vanilla vs. Hard Step)
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
python train_nam_threshold.py
```

---

## 🔍 Model Architecture Detail

The core innovation is implemented directly inside the custom sub-network modules via boolean tensor masking during the forward pass:

$$\mathbf{1}\{x < 0.7\}, \quad \mathbf{1}\{0.7 \le x < 1.0\}, \quad \mathbf{1}\{x \ge 1.0\}$$

This mathematically limits the optimization landscape to a three-level step function. Backpropagation updates only the distinct active coefficient block $b_k$ mapping to the patient's incoming real-time telemetry slice, guaranteeing zero slope leakage inside safe clinical bands.

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
