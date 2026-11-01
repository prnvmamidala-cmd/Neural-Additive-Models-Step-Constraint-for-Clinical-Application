"""eICU-CRD Demo v2.0.1 (open, ODbL). One row per hospital stay (first ICU stay, unitvisitnumber==1).
SBP = first SBP in first 6 h of the ICU stay (non-invasive aperiodic preferred, else invasive periodic);
HR = periodic heart rate closest in time (within 15 min) to that SBP. Outcome = hospital death."""
import numpy as np, pandas as pd, os
import os; D=os.environ.get("EICU_DIR","data_pn/eicu")
p=pd.read_csv(f"{D}/patient.csv.gz")
p=p[(p.unitvisitnumber==1)&p.hospitaldischargestatus.isin(["Alive","Expired"])].copy()
p["age"]=pd.to_numeric(p.age.replace("> 89","90"),errors="coerce"); p=p[p.age>=18]
vp=pd.read_csv(f"{D}/vitalPeriodic.csv.gz",usecols=["patientunitstayid","observationoffset","heartrate","systemicsystolic"])
va=pd.read_csv(f"{D}/vitalAperiodic.csv.gz",usecols=["patientunitstayid","observationoffset","noninvasivesystolic"])
w=lambda d:d[(d.observationoffset>=0)&(d.observationoffset<=360)]
vp,va=w(vp),w(va)
s1=va[va.noninvasivesystolic.between(40,300)].rename(columns={"noninvasivesystolic":"sbp"})[["patientunitstayid","observationoffset","sbp"]]
s2=vp[vp.systemicsystolic.between(40,300)].rename(columns={"systemicsystolic":"sbp"})[["patientunitstayid","observationoffset","sbp"]]
s1=s1.sort_values("observationoffset").groupby("patientunitstayid").first()
s2=s2.sort_values("observationoffset").groupby("patientunitstayid").first()
sb=s1.combine_first(s2).reset_index()
hr=vp[vp.heartrate.between(20,250)][["patientunitstayid","observationoffset","heartrate"]]
m=sb.merge(hr,on="patientunitstayid",suffixes=("","_hr")); m["dt"]=(m.observationoffset_hr-m.observationoffset).abs()
m=m[m.dt<=15].sort_values("dt").groupby("patientunitstayid").first().reset_index()
t=p.merge(m,on="patientunitstayid")
t=pd.DataFrame(dict(id=t.patientunitstayid,age=t.age,hr=t.heartrate,sbp=t.sbp,y=(t.hospitaldischargestatus=="Expired").astype(float)))
t["si"]=t.hr/t.sbp
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data_local")
os.makedirs(out, exist_ok=True)
t.to_csv(os.path.join(out,"eicu_demo_si_age_mortality.csv"),index=False)
print(len(t),t.y.mean(),(t.si<0.7).sum(),((t.si>=0.7)&(t.si<1)).sum(),(t.si>=1).sum())
print(t.groupby(pd.cut(t.si,[0,0.7,1.0,9])).y.agg(["mean","size"]))
