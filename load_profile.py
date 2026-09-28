"""
단계 2 — 부하 곡선 두 종류 (시간 단위, 2023~2025)

etth2_pu : ETTh2 실측 부하의 '모양'. 이상 구간 대체 후 상위 1% 값 = 1.0
dc_pu    : 데이터센터형. IT 일정 × PUE(외기). 상위 1% 값 = 1.0

실험에서는 여기에 '목표 부하율'을 곱해 MVA로 바꾼다.
예) 목표 0.5 → 상위 1% 시간대 부하가 정격의 50%

판단 근거는 config.yaml load_profile 절에 있다.
"""
import numpy as np
import pandas as pd
from thermal_model import load_config

cfg = load_config()
lp = cfg["load_profile"]
amb = pd.read_csv(cfg["weather"]["file"], parse_dates=["time"], index_col="time")["temp_c"]
target = amb.index                                   # 2023-01-01 00시 ~ 2025-12-31 23시


# ── ETTh2 ────────────────────────────────────────────────
e = lp["etth2"]
src = pd.read_csv(e["source"], parse_dates=["date"], index_col="date")[e["column"]].astype(float)

bad = src <= 0
bad |= (src.index >= e["event_start"]) & (src.index <= e["event_end"])
clean = src.copy()
clean[bad] = np.nan
for lag in (7, 14, 21):                              # 7일 전 → 없으면 14일 전 → 21일 전
    fill = clean.shift(freq=pd.Timedelta(days=lag)).reindex(clean.index)
    clean = clean.fillna(fill)
assert clean.isna().sum() == 0, "대체 못 한 구간 있음"
print(f"ETTh2 이상 구간 대체: {int(bad.sum())}시간")

# 같은 월·일·시로 옮기기
def source_time(t):
    # 2023·2025 → 주기 A(2016.07~2017.06), 2024 → 주기 B(2017.07~2018.06)
    base = 2016 if t.year in (2023, 2025) else 2017
    y = base if t.month >= 7 else base + 1
    d = min(t.day, 28) if (t.month == 2 and t.day == 29) else t.day
    s = pd.Timestamp(year=y, month=t.month, day=d, hour=t.hour)
    if s > clean.index[-1]:                          # 2018-06-27 이후는 원본에 없음 → 주기 A의 같은 날
        s = s.replace(year=2017)
    return s

src_idx = [source_time(t) for t in target]
etth2 = pd.Series(clean.reindex(src_idx).values, index=target)
etth2_replaced = pd.Series(bad.reindex(src_idx).values, index=target)
assert etth2.isna().sum() == 0

# ── 데이터센터형 ───────────────────────────────────────────
d = lp["datacenter"]
frac = ((amb - d["t_low_c"]) / (d["t_high_c"] - d["t_low_c"])).clip(0, 1)
dc = d["pue_low"] + (d["pue_high"] - d["pue_low"]) * frac   # IT = 1 로 두고 PUE를 곱함

# ── 정규화: 상위 1% 값 = 1.0 ─────────────────────────────────
def to_pu(x):
    return x / x.quantile(0.99)

out = pd.DataFrame({
    "etth2_pu": to_pu(etth2),
    "dc_pu": to_pu(dc),
    "etth2_replaced": etth2_replaced,
})
out.index.name = "time"
out.to_csv(lp["file"])

print(out[["etth2_pu", "dc_pu"]].describe().round(3))
print("외기와의 상관:", {c: round(out[c].corr(amb), 2) for c in ["etth2_pu", "dc_pu"]})
