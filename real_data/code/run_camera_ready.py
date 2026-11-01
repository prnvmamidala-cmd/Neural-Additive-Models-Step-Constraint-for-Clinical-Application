#!/usr/bin/env python3
"""Camera-ready experiments. Usage: python run_camera_ready.py <nhanes|physionet2012|eicu-demo>
For each seed s in 0..4: one stratified 80/20 train/test split (random_state=s); all models share it, so
differences are paired. NAMs: Vanilla, Hard-Step (0.7,1.0), plus cutoff sensitivity (0.6,0.9), (0.8,1.1).
Baselines: binned LR (one-hot SI bins + linear age) and binned LR + cubic age spline (5 knots), L2 C=1.
No validation-based selection: hyperparameters are fixed a priori; the 20% split is the held-out test set."""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import nam_models as M

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
FILES = {"nhanes": "nhanes_si_age_cvd.csv", "physionet2012": "physionet2012_si_age_mortality.csv", "eicu-demo": "eicu_demo_si_age_mortality.csv"}
CUTS = [(0.6, 0.9), (0.7, 1.0), (0.8, 1.1)]
DS = sys.argv[1]; SEEDS = range(5)
df = pd.read_csv(ROOT / "data_local" / FILES[DS])
si, age, y = (df[c].to_numpy(np.float32) for c in ["si", "age", "y"])
T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()

def onehot(x, t=(0.7, 1.0)):
    return np.stack([x < t[0], (x >= t[0]) & (x < t[1]), x >= t[1]], 1).astype(float)

def binned_lr(tr, te, spline):
    sc = StandardScaler().fit(age[tr, None])
    if spline:
        sp = SplineTransformer(n_knots=5, degree=3).fit(age[tr, None]); A = lambda i: sp.transform(age[i, None])
    else:
        A = lambda i: sc.transform(age[i, None])
    X = lambda i: np.hstack([onehot(si[i])[:, 1:], A(i)])  # reference bin SI<0.7
    m = LogisticRegression(C=1.0, max_iter=5000).fit(X(tr), y[tr]); return m.predict_proba(X(te))[:, 1]

def run(seed):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y.astype(int))
    tsi, tag, ty = T(si[tr]).view(-1, 1), T(age[tr]).view(-1, 1), T(y[tr])
    esi, eag, ey = T(si[te]).view(-1, 1), T(age[te]).view(-1, 1), T(y[te])
    M.set_seed(seed); loader = DataLoader(TensorDataset(tsi, tag, ty), batch_size=M.BATCH, shuffle=True)
    out, keep = {"seed": seed, "n_train": len(tr), "n_test": len(te)}, {}
    models = [("vanilla", M.VanillaNAM)] + [(f"step_{a}_{b}", (lambda t=(a, b): M.StepNAM(t))) for a, b in [CUTS[1], CUTS[0], CUTS[2]]]
    for name, ctor in models:
        M.set_seed(seed); m = ctor(); h = M.train_model(m, loader, esi, eag, ey); p = M.predict(m, esi, eag)
        r = dict(auroc=float(roc_auc_score(y[te], p)), brier=float(brier_score_loss(y[te], p)), test_bce=h["test_bce"][-1])
        if name.startswith("step"):
            f = m.f_si; b = [float(f.b0), float(f.b1), float(f.b2)]; r.update(b=b, monotone=b[0] < b[1] < b[2])
        out[name] = r; keep[name] = (m, h, p)
    for name, spl in [("binned_lr", False), ("binned_lr_spline", True)]:
        p = binned_lr(tr, te, spl); out[name] = dict(auroc=float(roc_auc_score(y[te], p)), brier=float(brier_score_loss(y[te], p))); keep[name] = (None, None, p)
    if seed == 0:  # plot data from seed 0 split; binned rates only, no row-level records
        g = np.linspace(0.3, 1.6, 600).astype(np.float32); gt = T(g).view(-1, 1)
        with torch.no_grad():
            fv = keep["vanilla"][0].f_si(gt).numpy(); fs = keep["step_0.7_1.0"][0].f_si(gt).numpy()
        roc = {f"fpr_{k}": roc_curve(y[te], keep[k][2])[0] for k in ["vanilla", "step_0.7_1.0", "binned_lr_spline"]}
        roc.update({f"tpr_{k}": roc_curve(y[te], keep[k][2])[1] for k in ["vanilla", "step_0.7_1.0", "binned_lr_spline"]})
        edges = np.round(np.arange(0.3, 1.61, 0.1), 2)
        bin_n, bin_rate = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mk = (si >= lo) & (si < hi) if hi < 1.6 else si >= lo
            bin_n.append(int(mk.sum())); bin_rate.append(float(y[mk].mean()) if mk.sum() else np.nan)
        (ROOT / "results").mkdir(exist_ok=True)
        np.savez(ROOT / "results" / f"plotdata_{DS}.npz", grid=g, fv=fv - fv.mean(), fs=fs - fs.mean(),
                 bin_edges=edges, bin_n=np.array(bin_n), bin_rate=np.array(bin_rate),
                 **roc, **{f"{k}_{n}": np.array(keep[n][1][k]) for k in ["train_bce", "test_bce"] for n in ["vanilla", "step_0.7_1.0"]})
    return out

runs = [run(s) for s in SEEDS]
MODELS = ["vanilla", "step_0.7_1.0", "binned_lr", "binned_lr_spline", "step_0.6_0.9", "step_0.8_1.1"]
ms = lambda v: [float(np.mean(v)), float(np.std(v, ddof=1))]
summ = {m: {k: ms([r[m][k] for r in runs]) for k in ["auroc", "brier"]} for m in MODELS}
for m in MODELS:
    if m.startswith("step"):
        summ[m]["monotone_runs"] = f"{sum(r[m]['monotone'] for r in runs)}/{len(runs)}"
        summ[m]["b_mean_sd"] = [ms([r[m]["b"][i] for r in runs]) for i in range(3)]
paired = {}
for m in ["vanilla", "binned_lr", "binned_lr_spline"]:
    d = np.array([r[m]["auroc"] - r["step_0.7_1.0"]["auroc"] for r in runs])
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    paired[f"{m}_minus_step"] = dict(mean=float(d.mean()), sd=float(d.std(ddof=1)), ci95_t=[float(d.mean() - h), float(d.mean() + h)], per_seed=d.tolist())
bins = {f"{a}_{b}": onehot(si, (a, b)).sum(0).astype(int).tolist() for a, b in CUTS}
rates = {f"{a}_{b}": [float(y[onehot(si, (a, b))[:, i] == 1].mean()) if onehot(si, (a, b))[:, i].sum() else None for i in range(3)] for a, b in CUTS}
met = dict(dataset=DS, n=len(y), events=int(y.sum()), prevalence=float(y.mean()), si_bin_counts=bins, event_rate_by_bin=rates,
           summary_mean_sd=summ, paired_auroc_diff_vs_step=paired, runs=runs,
           protocol="seeds 0-4; each seed = its own stratified 80/20 train/test split + init; fixed 150 epochs, Adam 1e-3, batch 64, BCE; "
                    "no early stopping or selection on test; LR baselines: L2 C=1, spline n_knots=5 degree=3 on age; not survey weighted")
(ROOT / "results").mkdir(exist_ok=True)
json.dump(met, open(ROOT / "results" / f"metrics_{DS}.json", "w"), indent=1)
print(json.dumps({k: met[k] for k in ["n", "events", "summary_mean_sd", "paired_auroc_diff_vs_step", "si_bin_counts"]}, indent=1))
