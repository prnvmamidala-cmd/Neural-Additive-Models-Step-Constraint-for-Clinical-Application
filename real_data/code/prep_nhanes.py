"""NHANES 2017-Mar 2020 Pre-Pandemic: P_DEMO, P_BPXO, P_MCQ (.xpt from CDC). Adults >=20.
SI = BPXOPLS1/BPXOSY1; y = any self-reported CHF/CHD/angina/MI/stroke (MCQ160B-F == 1).
Usage: NHANES_DIR=<dir with xpt> python prep_nhanes.py  -> data_local/nhanes_si_age_cvd.csv"""
import os, numpy as np, pandas as pd, pyreadstat
from pathlib import Path
D = Path(os.environ.get("NHANES_DIR", "data_nhanes"))
rx = lambda n: pyreadstat.read_xport(str(D / f"{n}.xpt"))[0]
df = rx("P_DEMO")[["SEQN", "RIDAGEYR"]].merge(rx("P_BPXO")[["SEQN", "BPXOSY1", "BPXOPLS1"]], on="SEQN") \
    .merge(rx("P_MCQ")[["SEQN"] + [f"MCQ160{c}" for c in "BCDEF"]], on="SEQN")
df = df[df.RIDAGEYR >= 20].copy(); cols = [f"MCQ160{c}" for c in "BCDEF"]
for c in cols: df.loc[~df[c].isin([1, 2]), c] = np.nan
df["y"] = np.where(df[cols].notna().any(axis=1), (df[cols] == 1).any(axis=1).astype(float), np.nan)
df["age"], df["sbp"], df["hr"] = df.RIDAGEYR.astype(float), df.BPXOSY1.astype(float), df.BPXOPLS1.astype(float)
df = df.dropna(subset=["sbp", "hr", "age", "y"]); df = df[df.sbp > 0].copy()
df["si"] = (df.hr / df.sbp).astype(np.float32); df["age"] = df.age.astype(np.float32); df["y"] = df.y.astype(np.float32)
out = Path(__file__).resolve().parent.parent / "data_local"
out.mkdir(parents=True, exist_ok=True)
df[["SEQN", "age", "hr", "sbp", "si", "y"]].reset_index(drop=True).to_csv(out / "nhanes_si_age_cvd.csv", index=False)
print(len(df), df.y.mean())
