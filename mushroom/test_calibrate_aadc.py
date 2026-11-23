"""Adjoint-vs-finite-difference check for calibrate_aadc (the only place finite differences are used). Offline."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aadc  # noqa: E402
from mushroom import calibrate_aadc as C  # noqa: E402


def synthetic_cases(n=6, days=8, seed=1):
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n):
        R = [list(rng.gamma(0.5, 4.0, C.J)) for _ in range(days)]
        T = [list(12 + 4 * rng.standard_normal(C.J)) for _ in range(days)]
        cases.append({"gbif": i, "year": 2020, "R": R, "T": T, "WE": [float(d % 7 >= 5) for d in range(days)]})
    return cases


def test_adjoint_matches_fd_and_mirror():
    cases = synthetic_cases()
    pool = aadc.ThreadPool(8)
    f, h, out = C.record_loss(cases)
    x = C.X0 + np.array([0.1, -0.2, 0.7, 0.1, -0.3, 0.2, 0.05])

    def run(xx):
        v, d = aadc.evaluate(f, {out: h}, {hi: np.array([xi]) for hi, xi in zip(h, xx)}, pool)
        return float(v[out][0]), np.array([float(d[out][hi][0]) for hi in h])

    val, grad = run(x)
    # cohort mirror: the same loss in plain floats on a separate code path
    th = dict(zip(C.PARAMS, x))
    def aw(lt):
        q = math.exp(-math.exp(-lt)); raw = [q ** j for j in range(C.J)]; s = sum(raw); return [r / s for r in raw]
    aR, aT = aw(th["log_tau_r"]), aw(th["log_tau_t"])
    ref = 0.0
    for c in cases:
        s = []
        for rr, tt, we in zip(c["R"], c["T"], c["WE"]):
            R = sum(a * r for a, r in zip(aR, rr)); Tm = sum(a * t for a, t in zip(aT, tt))
            s.append(th["w_r"] * R + th["w_t"] * math.exp(-(Tm - th["T_opt"]) ** 2 / (2 * math.exp(2 * th["log_sd"]))) + th["w_we"] * we)
        ref += math.log(sum(math.exp(v - s[0]) for v in s))
    for n in C.PARAMS:
        m, sd = C.PRIORS[n]; ref += 0.5 * ((th[n] - m) / sd) ** 2
    assert abs(val - ref) < 1e-9 * max(1.0, abs(ref))
    # adjoint vs central FD on the tape
    for k in range(len(x)):
        e = np.zeros_like(x); e[k] = 1e-5
        fd = (run(x + e)[0] - run(x - e)[0]) / 2e-5
        assert abs(fd - grad[k]) < 1e-4 * max(1.0, abs(grad[k])), (C.PARAMS[k], fd, grad[k])


def test_production_score_mirrors_the_tape():
    """scoring.fitted_score_raw (plain floats, live forecast) == the AADC tape's s(t) minus weekend."""
    import datetime
    from mushroom import scoring as S
    cases = synthetic_cases(n=2, days=3, seed=7)
    x = C.X0 + np.array([0.2, 0.4, 1.5, 0.2, 0.1, -0.2, 0.0])
    x[C.PARAMS.index("w_we")] = 0.0
    tape = C.scores_at(cases, x, aadc.ThreadPool(8))
    fp = {"params": dict(zip(C.PARAMS, map(float, x))), "J": C.J}
    for c, row in zip(cases, tape):
        for rr, tt, v in zip(c["R"], c["T"], row):
            # SiteWeather with the J lagged days in date order (oldest first), today last
            n = C.J
            dates = [datetime.date(2026, 1, 1) + datetime.timedelta(days=k) for k in range(n)]
            w = S.SiteWeather(ok=True, dates=dates, precip=list(rr[::-1]), precip_grid=list(rr[::-1]),
                              tmin=[t - 1.0 for t in tt[::-1]], tmax=[t + 1.0 for t in tt[::-1]], rh_mean=[80.0] * n)
            assert abs(S.fitted_score_raw(fp, w, n - 1) - v) < 1e-9
