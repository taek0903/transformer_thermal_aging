"""단계 2 확인 그림 — 두 부하 곡선의 하루·계절 모양과 외기와의 관계"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from thermal_model import load_config

cfg = load_config()
lp = pd.read_csv(cfg["load_profile"]["file"], parse_dates=["time"], index_col="time")
amb = pd.read_csv(cfg["weather"]["file"], parse_dates=["time"], index_col="time")["temp_c"]

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for c, lab in [("etth2_pu", "ETTh2 shape"), ("dc_pu", "data center (IT x PUE)")]:
    ax[0].plot(lp[c].groupby(lp.index.hour).mean(), marker=".", label=lab)
    ax[1].plot(lp[c].groupby(lp.index.month).mean(), marker="o", label=lab)
ax[0].set_title("mean by hour of day"); ax[0].set_xlabel("hour"); ax[0].set_ylabel("load (p99 = 1.0)")
ax[1].set_title("mean by month"); ax[1].set_xlabel("month"); ax[0].legend()
a2 = ax[1].twinx(); a2.plot(amb.groupby(amb.index.month).mean(), color="gray", ls="--"); a2.set_ylabel("ambient ℃ (dashed)")
ax[2].scatter(amb, lp["etth2_pu"], s=1, alpha=0.15, label="ETTh2")
ax[2].scatter(amb, lp["dc_pu"], s=1, alpha=0.15, label="data center")
ax[2].set_xlabel("ambient ℃ (Icheon)"); ax[2].set_title("load vs ambient"); ax[2].legend(markerscale=8)
plt.tight_layout(); plt.savefig("figs/step2_load_profiles.png", dpi=130)
print("figs/step2_load_profiles.png")
