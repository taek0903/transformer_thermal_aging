"""
단계 1 — 상대 노화율과 누적 수명 소모 (일반 크라프트지)

V = 2 ** ((theta_h - 98) / 6)
  최고점 98℃에서 V=1. 6K 오를 때마다 2배, 6K 내려갈 때마다 절반.
L = sum(V * dt)
  "98℃로 계속 운전했을 때" 대비 몇 시간어치 수명을 썼는가.
"""
import numpy as np


def aging_rate(theta_h, cfg):
    ins = cfg["insulation"]
    return 2.0 ** ((np.asarray(theta_h) - ins["reference_hotspot_c"]) / ins["doubling_k"])


def loss_of_life_hours(theta_h, cfg):
    dt_h = cfg["simulation"]["dt_min"] / 60
    return float(np.sum(aging_rate(theta_h, cfg)) * dt_h)
