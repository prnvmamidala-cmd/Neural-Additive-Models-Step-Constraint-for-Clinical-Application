import sys, json, numpy as np, matplotlib
from pathlib import Path
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Nimbus Roman","DejaVu Serif"],"font.size":8,
 "axes.titlesize":8.5,"axes.labelsize":8,"legend.fontsize":6.8,"xtick.labelsize":7,"ytick.labelsize":7,
 "axes.spines.top":False,"axes.spines.right":False,"lines.linewidth":1.4})
R=Path(__file__).resolve().parent.parent; O=str(R/"figures")+"/"
DS=[("NHANES (CVD history)","nhanes"),("PhysioNet 2012 (ICU mortality)","physionet2012"),("eICU Demo (hospital mortality)","eicu-demo")]; CV,CS="#1f5fa8","#c0392b"
L=[(t,np.load(R/"results"/f"plotdata_{p}.npz"),json.load(open(R/"results"/f"metrics_{p}.json"))) for t,p in DS]; k=len(L)
def thr(a):
    for x in (0.7,1.0): a.axvline(x,color="0.55",ls="--",lw=0.8)
# ROC
f,ax=plt.subplots(1,k,figsize=(3.5,3.5/k+0.6),sharey=True)
for a,(t,d,m) in zip(ax,L):
    r0=m["runs"][0]
    a.plot(d["fpr_vanilla"],d["tpr_vanilla"],color=CV,label=f"Vanilla {r0['vanilla']['auroc']:.3f}")
    a.plot(d["fpr_step_0.7_1.0"],d["tpr_step_0.7_1.0"],color=CS,ls="--",label=f"Step {r0['step_0.7_1.0']['auroc']:.3f}")
    a.plot(d["fpr_binned_lr_spline"],d["tpr_binned_lr_spline"],color="0.35",ls=":",lw=1.2,label=f"Bin LR+spl {r0['binned_lr_spline']['auroc']:.3f}")
    a.plot([0,1],[0,1],color="0.75",lw=0.7); a.set_title(t.split(" (")[0]); a.set_xlabel("FPR"); a.legend(loc="lower right",frameon=False,handlelength=1.2,fontsize=5.2,borderaxespad=0.1)
    a.set_aspect("equal")
ax[0].set_ylabel("TPR"); f.tight_layout(pad=0.3); f.savefig(O+"fig_roc.png",dpi=300); plt.close(f)
# event rate
f,ax=plt.subplots(1,k,figsize=(3.5,1.75),sharey=False)
for a,(t,d,m) in zip(ax,L):
    edges=np.asarray(d["bin_edges"]); c=0.5*(edges[:-1]+edges[1:]); n=np.asarray(d["bin_n"]); r=np.where(n>=15, np.asarray(d["bin_rate"], dtype=float), np.nan)
    a.bar(c,r,width=0.085,color=["#9db9d9" if x<0.7 else "#f0b27a" if x<1.0 else "#d9776b" for x in c],edgecolor="0.3",lw=0.3)
    thr(a); a.set_title(t.split(" (")[0]); a.set_xlabel("Shock Index"); a.set_xlim(0.3,1.6)
ax[0].set_ylabel("Observed event rate"); f.tight_layout(pad=0.3); f.savefig(O+"fig_data_event_rate.png",dpi=300); plt.close(f)
# shapes
f,ax=plt.subplots(1,k,figsize=(7.16,2.1))
for a,(t,d,m) in zip(ax,L):
    g=d["grid"]; a.plot(g,d["fv"],color=CV,label="Vanilla NAM (Tanh)"); a.plot(g,d["fs"],color=CS,lw=1.8,label="Hard step NAM")
    thr(a); b=m["runs"][0]["step_0.7_1.0"]["b"]; a.set_title(t)
    a.text(0.97,0.05,f"$b_0$={b[0]:.2f}, $b_1$={b[1]:.2f}, $b_2$={b[2]:.2f}",transform=a.transAxes,va="bottom",ha="right",fontsize=6.8)
    a.set_xlabel("Shock Index (HR/SBP)"); a.set_xlim(0.3,1.6)
ax[0].set_ylabel("Centered $f_{SI}(x)$"); h,l=ax[0].get_legend_handles_labels(); f.legend(h,l,loc="upper center",ncol=2,frameon=False,bbox_to_anchor=(0.5,1.02))
f.tight_layout(pad=0.3,rect=(0,0,1,0.92)); f.savefig(O+"clinical_step_activation_results.png",dpi=300); plt.close(f)
# training
f,ax=plt.subplots(1,k,figsize=(7.16,1.9))
for a,(t,d,m) in zip(ax,L):
    e=np.arange(1,151)
    for nm,c in (("vanilla",CV),("step",CS)):
        lab="Vanilla" if nm=="vanilla" else "Hard step"
        a.plot(e,d[f"train_bce_{nm if nm=='vanilla' else 'step_0.7_1.0'}"],color=c,lw=1,alpha=0.55,ls=":",label=f"{lab} train"); a.plot(e,d[f"test_bce_{nm if nm=='vanilla' else 'step_0.7_1.0'}"],color=c,lw=1.2,label=f"{lab} test")
    a.set_title(t); a.set_xlabel("Epoch")
ax[0].set_ylabel("BCE"); ax[-1].legend(loc="upper right",frameon=False,ncol=2)
f.tight_layout(pad=0.3); f.savefig(O+"fig_training_curves.png",dpi=300); plt.close(f)
print("ok")
