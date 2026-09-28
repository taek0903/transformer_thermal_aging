"""
단계 1 — IEC 60076-7 차분방정식 열모델 (+ 냉각팬 운전 로직)

입력 (1분 간격 배열)
  theta_a  : 주위온도 [℃]
  load_mva : 부하 [MVA]
  fan_ok   : 팬이 정상인가 (False면 팬을 켜야 해도 못 켬 — 단계 4 고장 주입용)
출력
  theta_o  : 최상부 유온 [℃]
  theta_h  : 권선 최고점 온도 [℃]
  fan_on   : 팬 운전 여부
  K        : 그 순간 냉각방식 기준 부하율

판단
- 팬이 꺼져 있으면 ONAN 파라미터를, 켜져 있으면 ONAF 파라미터를 쓴다.
  부하율 K도 그 냉각방식의 정격 기준으로 다시 계산한다.
  (팬 없이는 정격이 onan_to_onaf_ratio 만큼으로 줄어든다)
- 팬은 유온 기준으로 켜고 끈다. 켜는 값과 끄는 값 사이에 간격을 둬서
  경계에서 켜졌다 꺼졌다를 반복하지 않게 한다.
- 상태변수(유온, 최고점 두 성분)는 냉각방식이 바뀌어도 이어진다. 바뀌는 것은 목표값과 시정수뿐.
"""
import numpy as np
import yaml


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def steady_state(theta_a, K, p):
    """주어진 외기·부하율에서 충분히 오래 지난 뒤의 온도 (시작값·검증용)"""
    d_o = ((1 + K**2 * p["R"]) / (1 + p["R"])) ** p["x"] * p["d_theta_or"]
    d_h1 = p["k21"] * K ** p["y"] * p["d_theta_hr"]
    d_h2 = (p["k21"] - 1) * K ** p["y"] * p["d_theta_hr"]
    return theta_a + d_o, d_h1, d_h2


def simulate(theta_a, load_mva, cfg, fan_ok=None, fan_mode="auto"):
    """
    fan_mode: "auto" = 유온 기준 자동 / "ONAN" = 항상 꺼짐 / "ONAF" = 항상 켜짐
    """
    n = len(theta_a)
    if fan_ok is None:
        fan_ok = np.ones(n, dtype=bool)

    tp = cfg["thermal_params"]
    tr = cfg["transformer"]
    fc = cfg["fan_control"]
    dt = cfg["simulation"]["dt_min"]
    s_onaf = tr["rating_onaf_mva"]
    s_onan = s_onaf * tr["onan_to_onaf_ratio"]

    # 파이썬 반복문 속도를 위해 float로 꺼내둔다
    th_a = np.asarray(theta_a, dtype=float).tolist()
    load = np.asarray(load_mva, dtype=float).tolist()
    ok = np.asarray(fan_ok, dtype=bool).tolist()

    out_o = np.empty(n); out_h = np.empty(n); out_fan = np.empty(n, dtype=bool); out_K = np.empty(n)

    # 시작 상태: 첫 입력에서의 정상상태, 팬은 그 유온으로 판단
    fan = fan_mode == "ONAF"
    p = tp["ONAF"] if fan else tp["ONAN"]
    K = load[0] / (s_onaf if fan else s_onan)
    th_o, h1, h2 = steady_state(th_a[0], K, p)
    if fan_mode == "auto":
        fan = th_o >= fc["on_c"] and ok[0]

    for i in range(n):
        # 1) 팬 상태 결정
        if fan_mode == "auto":
            if fan and th_o <= fc["off_c"]:
                fan = False
            elif not fan and th_o >= fc["on_c"]:
                fan = True
            fan_now = fan and ok[i]          # 켜라는 신호가 있어도 고장이면 안 돈다
        else:
            fan_now = (fan_mode == "ONAF") and ok[i]

        p = tp["ONAF"] if fan_now else tp["ONAN"]
        K = load[i] / (s_onaf if fan_now else s_onan)
        Ky = K ** p["y"]

        # 2) IEC 60076-7 차분방정식 (한 스텝 = dt분)
        target_o = ((1 + K * K * p["R"]) / (1 + p["R"])) ** p["x"] * p["d_theta_or"]
        th_o += dt / (p["k11"] * p["tau_o_min"]) * (target_o - (th_o - th_a[i]))
        h1 += dt / (p["k22"] * p["tau_w_min"]) * (p["k21"] * p["d_theta_hr"] * Ky - h1)
        h2 += dt / (p["tau_o_min"] / p["k22"]) * ((p["k21"] - 1) * p["d_theta_hr"] * Ky - h2)

        out_o[i] = th_o
        out_h[i] = th_o + h1 - h2
        out_fan[i] = fan_now
        out_K[i] = K

    return {"theta_o": out_o, "theta_h": out_h, "fan_on": out_fan, "K": out_K}
