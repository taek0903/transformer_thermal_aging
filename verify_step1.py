"""
단계 1 검증 — 열모델이 규격이 말하는 대로 움직이는지 확인한다.
여기서 틀리면 뒤 실험은 전부 의미가 없다.

T1 정격 정상상태 (ONAF)   : K=1, θa=20℃ → θo≈72, θh≈98, V≈1
T2 정격 정상상태 (ONAN)   : 팬 없이 ONAN 정격 부하 → θh≈98
T3 유온 응답 속도          : 부하 계단 변화 후 63% 도달 시간 ≈ k11·τo
T4 최고점 과도응답         : 부하를 갑자기 올리면 최고점-유온 온도차가 최종값을 잠시 넘는가 (k21>1일 때 규격 모델의 특징)
T5 계산 간격 민감도        : dt 1분 vs 0.5분 결과 차이
T6 팬 히스테리시스         : 부하를 천천히 올렸다 내릴 때 팬이 65℃에서 켜지고 55℃에서 꺼지는가
T7 3년 실측 외기 실행      : 계산 시간, 일정 부하에서 연도별 수명 소모
"""
import copy, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from thermal_model import load_config, simulate
from aging import aging_rate, loss_of_life_hours

cfg = load_config()
S = cfg["transformer"]["rating_onaf_mva"]
S_ONAN = S * cfg["transformer"]["onan_to_onaf_ratio"]
MIN_PER_DAY = 1440
line = "-" * 60

# T1
n = 2 * MIN_PER_DAY
r = simulate(np.full(n, 20.0), np.full(n, S), cfg, fan_mode="ONAF")
print(line); print("T1 정격 정상상태 ONAF")
print(f"  θo={r['theta_o'][-1]:.2f}℃ (기대 72)  θh={r['theta_h'][-1]:.2f}℃ (기대 98)  V={aging_rate(r['theta_h'][-1], cfg):.3f} (기대 1)")

# T2
r = simulate(np.full(n, 20.0), np.full(n, S_ONAN), cfg, fan_mode="ONAN")
print(line); print("T2 정격 정상상태 ONAN (팬 없이 ONAN 정격 부하)")
print(f"  θo={r['theta_o'][-1]:.2f}℃  θh={r['theta_h'][-1]:.2f}℃  V={aging_rate(r['theta_h'][-1], cfg):.3f}")

# T3 / T4 — 0.5 → 1.0 계단 (ONAF 고정)
pre, post = 600, 1800
load = np.r_[np.full(pre, 0.5 * S), np.full(post, 1.0 * S)]
r3 = simulate(np.full(pre + post, 20.0), load, cfg, fan_mode="ONAF")
o = r3["theta_o"]; o0, o1 = o[pre - 1], o[-1]
t63 = np.argmax(o[pre:] >= o0 + 0.632 * (o1 - o0)) + 1
p = cfg["thermal_params"]["ONAF"]
print(line); print("T3 유온 응답 속도 (부하 0.5→1.0, ONAF)")
print(f"  63% 도달 {t63}분 (기대 k11·τo = {p['k11'] * p['tau_o_min']:.0f}분)")
h = r3["theta_h"]
g = h - r3["theta_o"]                     # 최고점-유온 온도차 (권선 쪽 성분)
gpk = g[pre:].max(); gfin = g[-1]
print(line); print("T4 최고점 과도응답 — 최고점-유온 온도차 기준")
print(f"  온도차 최대 {gpk:.2f}K / 최종 {gfin:.2f}K / 넘친 폭 {gpk - gfin:+.2f}K, 최대 시점 계단 후 {np.argmax(g[pre:])}분")
print(f"  (최고점 온도 자체는 유온이 천천히 따라 올라와서 최종값을 넘지 않음: 최대 {h[pre:].max():.2f}℃ / 최종 {h[-1]:.2f}℃)")

# T5
cfg2 = copy.deepcopy(cfg); cfg2["simulation"]["dt_min"] = 0.5
load2 = np.repeat(load, 2)
r5 = simulate(np.full(len(load2), 20.0), load2, cfg2, fan_mode="ONAF")
diff = np.max(np.abs(r5["theta_h"][1::2] - h))
print(line); print("T5 계산 간격 민감도 (dt 1분 vs 0.5분)")
print(f"  최고점 최대 차이 {diff:.3f}K")

# T6 — 12시간 동안 0→1.2 상승, 12시간 동안 1.2→0 하강 (자동 팬)
half = 12 * 60
up = np.linspace(0, 1.2 * S, half); ramp = np.r_[up, up[::-1], np.zeros(6 * 60)]
r6 = simulate(np.full(len(ramp), 25.0), ramp, cfg, fan_mode="auto")
f = r6["fan_on"].astype(int); sw = np.flatnonzero(np.diff(f)) + 1
print(line); print("T6 팬 히스테리시스 (외기 25℃, 부하 0→1.2→0)")
print(f"  전환 횟수 {len(sw)}")
for i in sw:
    print(f"   {i:5d}분  {'켜짐' if f[i] else '꺼짐'}  유온 {r6['theta_o'][i-1]:.2f}℃  부하 {ramp[i]/S:.2f}(ONAF 기준)")

# T7 — 실측 외기 3년, 부하 일정 0.7 (ONAF 기준)
amb = pd.read_csv(cfg["weather"]["file"], parse_dates=["time"], index_col="time")["temp_c"]
amb_min = amb.resample("1min").interpolate()          # 1시간 → 1분 (직선)
t0 = time.time()
r7 = simulate(amb_min.values, np.full(len(amb_min), 0.7 * S), cfg, fan_mode="auto")
el = time.time() - t0
res = pd.DataFrame({"h": r7["theta_h"], "o": r7["theta_o"], "fan": r7["fan_on"]}, index=amb_min.index)
print(line); print(f"T7 실측 외기 3년, 부하 0.7 일정 — {len(amb_min):,}스텝 {el:.1f}초")
for y, g in res.groupby(res.index.year):
    print(f"  {y}: 수명소모 {loss_of_life_hours(g['h'].values, cfg):7.1f}h (연 8,760h 대비)  최고점 최대 {g['h'].max():.1f}℃  팬 가동 {g['fan'].mean()*100:4.1f}%")

# 그림
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
tt = np.arange(-pre, post)
ax[0].plot(tt, r3["theta_o"], label="top-oil θo"); ax[0].plot(tt, r3["theta_h"], label="hot-spot θh")
ax[0].axvline(0, color="gray", lw=0.8); ax[0].set_xlabel("minutes from load step (0.5→1.0 pu)"); ax[0].set_ylabel("℃")
ax[0].set_title("T3/T4 step response (ONAF)"); ax[0].legend()
t6 = np.arange(len(ramp)) / 60
ax[1].plot(t6, r6["theta_o"], label="top-oil θo"); ax[1].plot(t6, ramp / S * 50, label="load ×50 (pu)", alpha=0.6)
ax[1].fill_between(t6, 0, 100, where=r6["fan_on"], color="C2", alpha=0.12, label="fan on")
ax[1].axhline(65, ls="--", color="C3", lw=0.8); ax[1].axhline(55, ls="--", color="C0", lw=0.8)
ax[1].set_ylim(20, 100); ax[1].set_xlabel("hours"); ax[1].set_title("T6 fan hysteresis (on 65 / off 55)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("figs/step1_verify.png", dpi=130)
print(line); print("그림: figs/step1_verify.png")
