"""Build per-patient SI/Age/mortality table from PhysioNet/CinC Challenge 2012 (sets a,b,c; ODC-BY open).
SI = first time-paired HR / SBP within first 6 h of ICU stay (NISysABP preferred, else invasive SysABP).
Outcome = In-hospital_death. Vitals come from the first 6 h only, outcome occurs after, so no temporal leakage."""
import glob, os, numpy as np, pandas as pd
import os; D=os.environ.get("PN2012_DIR","data_pn"); rows=[]
for s in "abc":
    out=pd.read_csv(f"{D}/Outcomes-{s}.txt").set_index("RecordID")["In-hospital_death"]
    for fp in glob.glob(f"{D}/set-{s}/*.txt"):
        df=pd.read_csv(fp); rid=int(df.loc[df.Parameter=="RecordID","Value"].iloc[0])
        age=float(df.loc[df.Parameter=="Age","Value"].iloc[0])
        df["min"]=df.Time.str.split(":").apply(lambda t:int(t[0])*60+int(t[1]))
        df=df[(df["min"]<=360)]
        piv=df[df.Parameter.isin(["HR","NISysABP","SysABP"])].pivot_table(index="min",columns="Parameter",values="Value",aggfunc="first")
        for c in ["HR","NISysABP","SysABP"]:
            if c not in piv: piv[c]=np.nan
        piv=piv[(piv.HR.between(20,250))]
        sbp=piv.NISysABP.where(piv.NISysABP.between(40,300)).fillna(piv.SysABP.where(piv.SysABP.between(40,300)))
        piv=piv.assign(sbp=sbp).dropna(subset=["sbp"])
        if len(piv)==0 or age<0 or rid not in out.index: continue
        r=piv.iloc[0]; rows.append(dict(rid=rid,set=s,age=age,hr=r.HR,sbp=r.sbp,y=float(out[rid])))
t=pd.DataFrame(rows); t["si"]=t.hr/t.sbp
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data_local")
os.makedirs(out, exist_ok=True)
t.to_csv(os.path.join(out,"physionet2012_si_age_mortality.csv"),index=False)
print(len(t), t.y.mean(), (t.si<0.7).sum(), ((t.si>=0.7)&(t.si<1)).sum(), (t.si>=1).sum())
print(t.groupby(pd.cut(t.si,[0,0.7,1.0,9])).y.mean())
