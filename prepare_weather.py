"""
단계 0 — 기상청 ASOS 이천(203) 시간 기온 정리

입력 : data/OBS_ASOS_TIM_*.csv  (기상청 원본, CP949)
출력 : data/ambient_icheon_2023_2025.csv   (시각, 외기온도, 보간 여부)
       data/ambient_gaps.csv                (보간한 시각 목록)

판단
- 기상청은 하루를 01~24시로 센다. 원본 '12-31 24:00'은 pandas에서 '다음날 00:00'으로 읽힌다.
  연도별 파일 경계가 겹치므로 중복 시각은 하나만 남긴다(값이 다르면 멈춘다).
- 진짜 결측(2024-08-06 10~11시, 2025-03-27 15시)은 앞뒤 값으로 직선 보간한다.
  변압기 유온은 시정수가 수 시간이라 1~2시간 보간의 영향은 작다.
  3시간 이상 연속 결측이면 보간하지 않고 멈춘다 — 그때는 따로 판단한다.
"""
import glob
import pandas as pd

START, END = "2023-01-01 00:00", "2025-12-31 23:00"
MAX_GAP_HOURS = 2

frames = []
for f in sorted(glob.glob("data/OBS_ASOS_TIM_*.csv")):
    d = pd.read_csv(f, encoding="cp949")
    d.columns = ["stn", "name", "time", "temp_c"]
    d["time"] = pd.to_datetime(d["time"])
    frames.append(d[["time", "temp_c"]])
raw = pd.concat(frames)

# 경계 중복 확인: 같은 시각에 값이 다르면 원본을 다시 봐야 한다
conflict = raw.groupby("time")["temp_c"].nunique().gt(1).sum()
assert conflict == 0, f"같은 시각에 다른 값 {conflict}건"
raw = raw.drop_duplicates("time").set_index("time").sort_index()

full = pd.date_range(START, END, freq="h")
s = raw["temp_c"].reindex(full)

# 연속 결측 길이 확인
gaps = s[s.isna()].index
runs = (pd.Series(gaps).diff() != pd.Timedelta("1h")).cumsum()
longest = pd.Series(gaps).groupby(runs).size().max() if len(gaps) else 0
assert longest <= MAX_GAP_HOURS, f"연속 결측 {longest}시간 — 보간 전에 판단 필요"

out = pd.DataFrame({"temp_c": s.interpolate(method="time"), "interpolated": s.isna()})
out.index.name = "time"
out.to_csv("data/ambient_icheon_2023_2025.csv")
out[out.interpolated].to_csv("data/ambient_gaps.csv")

print(f"행 {len(out)} / 기대 {len(full)}, 보간 {out.interpolated.sum()}시간, 최장 연속결측 {longest}시간")
print(out[out.interpolated])
print(out["temp_c"].groupby(out.index.year).describe().round(1))
