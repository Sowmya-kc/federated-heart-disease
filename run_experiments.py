"""
run_experiments.py -- runs every experiment in the project and writes tables + figures to results/.
    python run_experiments.py            (about 1-2 minutes)
All numbers are averaged over N_SEEDS different random train/test splits + initialisations,
because the test sets are small (~150 patients) and a single split can be misleading.
"""
import json, time, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
from data_utils import load_all, prepare_clients, missingness_report
from federated import (run_fedavg, run_centralized, run_local_only, eval_everywhere,
                       metrics, pool, HP)

N_SEEDS = 10
R = "results/"
sns.set_theme(style="whitegrid", context="talk", font_scale=0.7)
PAL = {"Local-only": "#d95f02", "Centralized": "#1b9e77", "FedAvg": "#2c5aa0",
       "FedProx": "#7570b3"}
t0 = time.time()
sites, info = load_all()
SITE = list(sites)

# ---------------- 0. dataset description ----------------
rows = []
for n, d in sites.items():
    rows.append(dict(site=n, patients=len(d), disease_pct=round(d.target.mean() * 100, 1),
                     mean_age=round(d.age.mean(), 1), male_pct=round(d.sex.mean() * 100, 1)))
pd.DataFrame(rows).to_csv(R + "01_dataset_summary.csv", index=False)
mr = missingness_report(sites); mr.to_csv(R + "02_missingness.csv")
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(x="site", y="disease_pct", hue="site", legend=False, data=pd.DataFrame(rows), ax=ax[0], palette=["#2c5aa0", "#1b9e77", "#d95f02"])
ax[0].set_title("Heart-disease prevalence per hospital"); ax[0].set_ylabel("% patients with disease"); ax[0].set_xlabel("")
sns.heatmap(mr, annot=True, fmt=".0f", cmap="Reds", cbar=False, ax=ax[1]); ax[1].set_title("% missing values per feature")
plt.tight_layout(); plt.savefig(R + "fig1_non_iid_data.png", dpi=160); plt.close()

# ---------------- 1. main comparison ----------------
agg, per_client, hist_f, hist_c, hist_p = [], [], [], [], []
best_rounds = {"FedAvg": [], "Centralized": []}
cm_f = np.zeros((2, 2)); roc_store = {"FedAvg": [], "Centralized": []}
for s in range(N_SEEDS):
    cl, cols = prepare_clients(sites, seed=s)
    gF, hF = run_fedavg(cl, seed=s)
    gP, hP = run_fedavg(cl, seed=s, mu=0.05)
    gC, hC = run_centralized(cl, seed=s)
    loc = run_local_only(cl, seed=s)
    Xte, yte = pool(cl, "te")
    best_rounds["FedAvg"].append(gF.best_round); best_rounds["Centralized"].append(gC.best_round)
    models = {"FedAvg": gF, "FedProx": gP, "Centralized": gC}
    for name, m in models.items():
        e = eval_everywhere(m, cl)
        agg.append(dict(seed=s, method=name, **e["Pooled"]))
        for n in SITE: per_client.append(dict(seed=s, method=name, site=n, **e[n]))
    # local-only: each hospital's own model, tested on (a) its own test set, (b) the pooled test set
    for n, m in loc.items():
        e = eval_everywhere(m, cl)
        per_client.append(dict(seed=s, method="Local-only", site=n, **e[n]))
        agg.append(dict(seed=s, method=f"Local-only ({n})", **e["Pooled"]))
    cm_f += confusion_matrix(yte, (gF.predict_proba(Xte) >= .5).astype(int))
    for k, m in (("FedAvg", gF), ("Centralized", gC)):
        roc_store[k].append(roc_curve(yte, m.predict_proba(Xte))[:2])
    hist_f.append(pd.DataFrame(hF)); hist_c.append(pd.DataFrame(hC)); hist_p.append(pd.DataFrame(hP))
    print(f"seed {s} done  ({time.time()-t0:.0f}s)")

A = pd.DataFrame(agg)
A.to_csv(R + "03_all_runs_pooled.csv", index=False)
A.loc[A.method.str.startswith("Local-only")].assign(method="Local-only (avg of 3)")
loc_avg = (A[A.method.str.startswith("Local-only")].groupby("seed").mean(numeric_only=True)
           .reset_index().assign(method="Local-only (avg of 3)"))
A2 = pd.concat([A[~A.method.str.startswith("Local-only")], loc_avg])
tab = A2.groupby("method")[["acc", "prec", "rec", "f1", "auc"]].agg(["mean", "std"]).round(3)
tab.columns = [f"{a}_{b}" for a, b in tab.columns]
order = ["Local-only (avg of 3)", "Centralized", "FedAvg", "FedProx"]
tab = tab.loc[order]; tab.to_csv(R + "04_main_comparison.csv")
print(tab[["acc_mean", "acc_std", "f1_mean", "auc_mean"]])

PC = pd.DataFrame(per_client); PC.to_csv(R + "05_per_client_all_runs.csv", index=False)
pct = PC.groupby(["site", "method"])[["acc", "f1", "auc"]].mean().round(3).reset_index()
pct.to_csv(R + "06_per_client_summary.csv", index=False)

# ---- fig2: convergence curves
HF = pd.concat(hist_f).groupby("round").agg(["mean", "std"]); HC = pd.concat(hist_c).groupby("round").agg(["mean", "std"])
HP_ = pd.concat(hist_p).groupby("round").agg(["mean", "std"])
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for H, lab in ((HC, "Centralized"), (HF, "FedAvg"), (HP_, "FedProx")):
    for a, k, yl in ((ax[0], "acc", "Pooled test accuracy"), (ax[1], "loss", "Pooled test loss")):
        a.plot(H.index, H[(k, "mean")], label=lab, color=PAL[lab], lw=2.2)
        a.fill_between(H.index, H[(k, "mean")] - H[(k, "std")], H[(k, "mean")] + H[(k, "std")], color=PAL[lab], alpha=.12)
        a.set_xlabel("Communication round (centralized: epochs / 3)"); a.set_ylabel(yl)
mF, mC = int(np.median(best_rounds["FedAvg"])), int(np.median(best_rounds["Centralized"]))
for a in ax:
    a.axvline(mF, color=PAL["FedAvg"], ls=":", lw=1.8)
    a.axvline(mC, color=PAL["Centralized"], ls=":", lw=1.8)
ax[1].annotate(f"early-stopping point\n(median: FedAvg r={mF}, central r={mC})", xy=(max(mF, mC), ax[1].get_ylim()[1] * .93), xytext=(max(mF, mC) + 4, ax[1].get_ylim()[1] * .93), fontsize=9, va="top")
ax[0].set_title("Convergence: accuracy"); ax[1].set_title("Convergence: loss (rises after early-stop point = overfitting)"); ax[0].legend()
plt.tight_layout(); plt.savefig(R + "fig2_convergence.png", dpi=160); plt.close()
HF.to_csv(R + "07_fedavg_round_history.csv")

# ---- fig3: main bar chart
fig, ax = plt.subplots(figsize=(10, 5))
m = A2.melt(id_vars=["seed", "method"], value_vars=["acc", "f1", "auc"], var_name="metric")
m["metric"] = m.metric.map({"acc": "Accuracy", "f1": "F1-score", "auc": "ROC-AUC"})
sns.barplot(data=m, x="metric", y="value", hue="method", hue_order=order, errorbar="sd", capsize=.08,
            palette=[PAL["Local-only"], PAL["Centralized"], PAL["FedAvg"], PAL["FedProx"]], ax=ax)
ax.set_ylim(0.5, 1); ax.set_xlabel(""); ax.set_ylabel(f"mean ± sd over {N_SEEDS} splits"); ax.legend(title="", loc="lower right")
ax.set_title("Pooled-test performance: Local vs Centralized vs Federated")
plt.tight_layout(); plt.savefig(R + "fig3_method_comparison.png", dpi=160); plt.close()

# ---- fig4: per client
fig, ax = plt.subplots(figsize=(11, 5))
sub = PC[PC.method.isin(["Local-only", "FedAvg", "Centralized"])]
sns.barplot(data=sub, x="site", y="acc", hue="method", hue_order=["Local-only", "Centralized", "FedAvg"], errorbar="sd", capsize=.08,
            palette=[PAL["Local-only"], PAL["Centralized"], PAL["FedAvg"]], ax=ax)
ax.set_ylim(0.5, 1); ax.set_xlabel(""); ax.set_ylabel("Accuracy on that hospital's own test set"); ax.legend(title="", loc="lower right")
ax.set_title("Client-wise accuracy")
plt.tight_layout(); plt.savefig(R + "fig4_per_client.png", dpi=160); plt.close()

# ---- fig5: confusion matrix + ROC
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(cm_f.astype(int), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax[0],
            xticklabels=["No disease", "Disease"], yticklabels=["No disease", "Disease"])
ax[0].set_title(f"FedAvg confusion matrix (summed over {N_SEEDS} splits)"); ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual")
grid = np.linspace(0, 1, 101)
for k in roc_store:
    tp = np.mean([np.interp(grid, f, t) for f, t in roc_store[k]], axis=0)
    ax[1].plot(grid, tp, label=f"{k} (AUC {tab.loc[k,'auc_mean']:.3f})", color=PAL[k], lw=2.2)
ax[1].plot([0, 1], [0, 1], "k--", lw=1); ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate")
ax[1].set_title("Mean ROC curve"); ax[1].legend()
plt.tight_layout(); plt.savefig(R + "fig5_confusion_roc.png", dpi=160); plt.close()

# ---------------- 2. communication overhead ----------------
cl0, _ = prepare_clients(sites, seed=0)
g0, _ = run_fedavg(cl0, seed=0, track=False)
n_par = g0.n_params
raw_bytes = sum(c["Xtr"].size * 8 for c in cl0.values())
tot_MB = HF[("MB", "mean")]
comm = dict(model_parameters=int(n_par), bytes_per_client_per_round_up=n_par * 8,
            total_MB_40_rounds=float(tot_MB.iloc[-1]), raw_training_data_MB=raw_bytes / 1e6)
json.dump(comm, open(R + "08_communication.json", "w"), indent=2)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(tot_MB.index, tot_MB.values, color=PAL["FedAvg"], lw=2.5); ax[0].set_xlabel("Round"); ax[0].set_ylabel("Cumulative traffic (MB)")
ax[0].set_title(f"Communication cost ({n_par} parameters, 3 clients)")
ax[1].plot(tot_MB.values, HF[("acc", "mean")].values, color=PAL["FedAvg"], lw=2.5); ax[1].set_xlabel("Cumulative traffic (MB)")
ax[1].set_ylabel("Pooled test accuracy"); ax[1].set_title("Accuracy per MB communicated")
plt.tight_layout(); plt.savefig(R + "fig6_communication.png", dpi=160); plt.close()

# ---------------- 3. local epochs / rounds trade-off ----------------
rows = []
for E in (1, 3, 5, 10):
    for s in range(N_SEEDS):
        cl, _ = prepare_clients(sites, seed=s)
        g, h = run_fedavg(cl, seed=s, local_epochs=E, rounds=40)
        e = eval_everywhere(g, cl)["Pooled"]
        r_to = next((x["round"] for x in h if x["acc"] >= 0.78), np.nan)
        rows.append(dict(local_epochs=E, seed=s, acc=e["acc"], auc=e["auc"], MB=h[-1]["MB"], rounds_to_78pct=r_to))
LE = pd.DataFrame(rows).groupby("local_epochs").mean().drop(columns="seed").round(3)
LE.to_csv(R + "09_local_epochs.csv")

# ---------------- 4. differential privacy ----------------
rows = []
for sig in (0, 0.05, 0.1, 0.2, 0.4, 0.8):
    for s in range(N_SEEDS):
        cl, _ = prepare_clients(sites, seed=s)
        g, _ = run_fedavg(cl, seed=s, dp_clip=1.0, dp_sigma=sig, track=False)
        e = eval_everywhere(g, cl)["Pooled"]; rows.append(dict(noise_sigma=sig, seed=s, acc=e["acc"], auc=e["auc"]))
DP = pd.DataFrame(rows); DP.to_csv(R + "10_dp_all.csv", index=False)
DPs = DP.groupby("noise_sigma")[["acc", "auc"]].agg(["mean", "std"]).round(3); DPs.to_csv(R + "10_dp_summary.csv")

# ---------------- 5. client dropout ----------------
rows = []
for p in (0, 0.2, 0.4, 0.6):
    for s in range(N_SEEDS):
        cl, _ = prepare_clients(sites, seed=s)
        g, _ = run_fedavg(cl, seed=s, dropout=p, track=False)
        e = eval_everywhere(g, cl)["Pooled"]; rows.append(dict(dropout_prob=p, seed=s, acc=e["acc"], auc=e["auc"]))
DO = pd.DataFrame(rows); DO.to_csv(R + "11_dropout_all.csv", index=False)
DOs = DO.groupby("dropout_prob")[["acc", "auc"]].agg(["mean", "std"]).round(3); DOs.to_csv(R + "11_dropout_summary.csv")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.lineplot(data=DP, x="noise_sigma", y="acc", marker="o", errorbar="sd", color=PAL["FedAvg"], ax=ax[0])
ax[0].axhline(tab.loc["FedAvg", "acc_mean"], color="gray", ls="--", label="FedAvg without DP"); ax[0].legend()
ax[0].set_title("Privacy–utility trade-off (clip + Gaussian noise)"); ax[0].set_xlabel("Noise multiplier σ"); ax[0].set_ylabel("Pooled accuracy")
sns.lineplot(data=DO, x="dropout_prob", y="acc", marker="o", errorbar="sd", color=PAL["FedProx"], ax=ax[1])
ax[1].set_title("Robustness to hospitals going offline"); ax[1].set_xlabel("Probability a hospital is offline in a round"); ax[1].set_ylabel("Pooled accuracy")
plt.tight_layout(); plt.savefig(R + "fig7_privacy_and_dropout.png", dpi=160); plt.close()

json.dump(dict(n_seeds=N_SEEDS, hyperparameters={k: list(v) if isinstance(v, tuple) else v for k, v in HP.items()},
               features=cols, usable_records={k: v[1] for k, v in info.items()}, early_stop_round_median={k: float(np.median(v)) for k, v in best_rounds.items()}), open(R + "00_config.json", "w"), indent=2)
print(f"ALL DONE in {time.time()-t0:.0f}s")
