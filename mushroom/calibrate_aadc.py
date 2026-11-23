#!/usr/bin/env python3
"""
Fit each species' fruiting timing to dated public finds, with AADC adjoint gradients.

MODEL_VERSION = "0.1.0"

Data: GBIF records in mainland Portugal (replay_gbif.gbif_records) and the Open-Meteo
historical archive at each record (cached in runs/gbif_weather_cache.json).

Design (matched case-control, the same days as replay_gbif.py): for each find, the
candidate days are the find day plus every 3rd day from -45 to +45 days, excluding
+/-7 days and days outside the species' season months. The model says which of those
days, at that same place, looks most like fruiting weather. A conditional logit
removes everything that does not change between those days (habitat, observer, place).

Score of day t:
    s(t) = w_r * R(t) + w_t * exp(-(T(t) - T_opt)^2 / (2 sd^2)) + w_we * weekend(t)
    R(t) = sum_j a_j(tau_r) * rain(t - j)      exponentially weighted rain, mm/day
    T(t) = sum_j a_j(tau_t) * tmean(t - j)     exponentially weighted mean temperature
    a_j(tau) = (1 - q) q^j / (1 - q^J),  q = exp(-1/tau),  j = 0..J-1
The window lengths tau_r and tau_t are fitted continuously, which is why this needs
derivatives through the weighting and not a grid search. weekend(t) absorbs the
observer effect (people record finds on weekends); it is a nuisance term, not biology.

Loss (MAP): -sum_i log softmax_i(find day) + weak priors (below). Gradient: AADC
classic 1.x adjoint, one reverse pass per optimiser step. No finite differences in
production; the finite-difference comparison lives in test_calibrate_aadc.py only.

Validation: fit on finds up to 2022, score finds from 2023 on, compare held-out AUC and
log-likelihood with the current hand-set model (replay_gbif scores on the same records)
and with chance. Then refit on all years for the parameters that are reported.

Usage (AADC venv):
  python mushroom/calibrate_aadc.py \
      [--species "Boletus edulis,..."] [--num-threads 8]
Writes runs/calibration_<today>.json.
"""

import argparse
import datetime
import json
import math
import os
import sys
import time
import urllib.error

import numpy as np
import aadc
from scipy.optimize import minimize

MUSHROOM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MUSHROOM_DIR))

from mushroom.replay_gbif import gbif_records, get, DEFAULT_SPECIES  # noqa: E402
from mushroom.species_db import SPECIES  # noqa: E402

MODEL_VERSION = "0.1.0"
J = 60                      # fixed — longest lag considered (days); the cited study's best windows were 20-32 d
PRE, POST = 110, 50         # fixed — archive days around each find: covers J days before the earliest control
CTRL_SPAN, CTRL_STEP, CTRL_GAP = 45, 3, 7   # identical to replay_gbif.py so the two are comparable
SPLIT_YEAR = 2022           # fixed — train <= 2022, test >= 2023
CACHE = os.path.join(MUSHROOM_DIR, "runs", "gbif_weather_cache.json")
FITTED_FILE = os.path.join(MUSHROOM_DIR, "fitted_params.json")   # read by scoring.py

PARAMS = ["w_r", "w_t", "T_opt", "log_sd", "log_tau_r", "log_tau_t", "w_we"]
# Weak priors (MAP), (mean, sd). Modelling choices, not measurements — fixed, stated here:
#   T_opt ~ N(13.7, 6): centred on the B. edulis optimum of Brejon Lamartiniere & Hoffman 2025
#     (bioRxiv 10.64898/2025.12.12.693895 v2), wide enough that any species can move off it;
#   log tau ~ N(log 20, 1): the same study's 20/26-day windows, a factor e either way at 1 sd;
#   weights ~ N(0, 3), log_sd ~ N(log 3, 1): only to keep the fit finite when data are thin.
PRIORS = {"w_r": (0.0, 3.0), "w_t": (0.0, 3.0), "T_opt": (13.7, 6.0), "log_sd": (math.log(3.0), 1.0),
          "log_tau_r": (math.log(20.0), 1.0), "log_tau_t": (math.log(20.0), 1.0), "w_we": (0.0, 3.0)}
# Box bounds (fixed — modelling choices): lag windows 3-45 days, inside the J=60 lag sum (the
# PT-only fit ran honey mushroom's temperature window to 125 days, i.e. off the end of the data);
# temperature spread 1-8 °C; T_opt 0-30 °C; weights unbounded.
BOUNDS = [(None, None), (None, None), (0.0, 30.0), (math.log(1.0), math.log(8.0)),
          (math.log(3.0), math.log(45.0)), (math.log(3.0), math.log(45.0)), (None, None)]
X0 = np.array([0.5, 1.0, 13.7, math.log(3.0), math.log(20.0), math.log(20.0), 0.0])


# ── data ──────────────────────────────────────────────────────────────

QUOTA = {"hit": False, "offline": False, "new": 0}


def save_cache(cache):
    tmp = CACHE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, CACHE)


def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            return json.load(f)
    return {}


def weather_for(rec, cache):
    k = str(rec["gbif"])
    if k in cache:
        return cache[k]
    d = datetime.date.fromisoformat(rec["date"])
    u = ("https://archive-api.open-meteo.com/v1/archive?"
         f"latitude={rec['lat']}&longitude={rec['lon']}"
         f"&start_date={d - datetime.timedelta(days=PRE)}&end_date={d + datetime.timedelta(days=POST)}"
         "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=UTC")
    if QUOTA["hit"] or QUOTA["offline"]:
        raise RuntimeError("weather not cached and fetching is off (quota hit or --offline)")
    try:
        dl = get(u)["daily"]
    except urllib.error.HTTPError as e:
        if e.code == 429:                                     # free-tier quota: stop fetching for this run
            QUOTA["hit"] = True
            print("  [quota] Open-Meteo 429: no more fetching this run; continuing with cached weather")
        raise
    w = {"start": dl["time"][0],
         "rain": [float(v) if v is not None else 0.0 for v in dl["precipitation_sum"]],
         "tmean": [0.5 * (a + b) if a is not None and b is not None else None
                   for a, b in zip(dl["temperature_2m_max"], dl["temperature_2m_min"])]}
    cache[k] = w
    QUOTA["new"] += 1
    if QUOTA["new"] % 25 == 0:                                # keep progress if the run is stopped
        save_cache(cache)
    time.sleep(1.0)                                           # paced: stay under the free per-minute cap
    return w


POOLS = {
    # Foragers rarely separate these four ceps (Boletus sect. Boletus) and photo IDs often cannot:
    # one forecast for "any porcino", fitted on all of their finds together.
    "Any porcino (Boletus sect. Boletus)": ["Boletus edulis", "Boletus reticulatus", "Boletus aereus"],
}


def build_cases(name, cache, countries=("PT",)):
    """One case per usable find: arrays of lagged rain/temp for each candidate day (find first)."""
    members = POOLS.get(name, [name])
    sps = [next(s for s in SPECIES if s.name == m) for m in members]
    season = sorted({mo for s in sps for mo in s.season})
    last_ok = datetime.date.today() - datetime.timedelta(days=7)
    cases = []
    recs = [r for m in members for c in countries for r in gbif_records(m, c)]
    for rec in recs:
        d = datetime.date.fromisoformat(rec["date"])
        if d + datetime.timedelta(days=POST) > last_ok:
            continue
        try:
            w = weather_for(rec, cache)
        except Exception as e:  # noqa: BLE001
            if not (QUOTA["hit"] or QUOTA["offline"]):
                print(f"  [warn] archive failed for {rec['gbif']}: {e}")
            continue
        start = datetime.date.fromisoformat(w["start"])
        days = [d] + [d + datetime.timedelta(days=k) for k in range(-CTRL_SPAN, CTRL_SPAN + 1, CTRL_STEP)
                      if abs(k) > CTRL_GAP and (d + datetime.timedelta(days=k)).month in season]
        R, T, WE, ok = [], [], [], True
        for t in days:
            i = (t - start).days
            if i - J + 1 < 0 or i >= len(w["rain"]):
                ok = False
                break
            rr = w["rain"][i - J + 1:i + 1][::-1]          # rr[j] = rain j days before t
            tt = w["tmean"][i - J + 1:i + 1][::-1]
            if any(v is None for v in tt):
                ok = False
                break
            R.append(rr)
            T.append(tt)
            WE.append(1.0 if t.weekday() >= 5 else 0.0)
        if ok and len(days) >= 4:
            cases.append({"gbif": rec["gbif"], "country": rec["country"], "year": d.year, "R": R, "T": T, "WE": WE})
    return cases


# ── the AADC tape ─────────────────────────────────────────────────────

def lag_weights(log_tau):
    """a_j(tau) as active values; normalised to sum to one over j = 0..J-1."""
    q = aadc.math.exp(-aadc.math.exp(-log_tau))           # q = exp(-1/tau)
    qj, raw = aadc.idouble(1.0), []
    for _ in range(J):
        raw.append(qj)
        qj = qj * q
    tot = raw[0]
    for v in raw[1:]:
        tot = tot + v
    return [v / tot for v in raw]


def day_scores(th, aR, aT, case):
    w_r, w_t, T_opt, log_sd, w_we = th["w_r"], th["w_t"], th["T_opt"], th["log_sd"], th["w_we"]
    two_var = 2.0 * aadc.math.exp(2.0 * log_sd)
    out = []
    for rr, tt, we in zip(case["R"], case["T"], case["WE"]):
        R = aR[0] * rr[0]
        Tm = aT[0] * tt[0]
        for j in range(1, J):
            R = R + aR[j] * rr[j]
            Tm = Tm + aT[j] * tt[j]
        dT = Tm - T_opt
        out.append(w_r * R + w_t * aadc.math.exp(-(dT * dT) / two_var) + w_we * we)
    return out


def record_loss(cases):
    """Tape: theta -> MAP loss over all cases. Returns (funcs, input handles, output handle)."""
    f = aadc.Functions()
    f.start_recording()
    th, handles = {}, []
    for n, v in zip(PARAMS, X0):
        th[n] = aadc.idouble(float(v))
        handles.append(th[n].mark_as_input())
    aR, aT = lag_weights(th["log_tau_r"]), lag_weights(th["log_tau_t"])
    loss = aadc.idouble(0.0)
    for c in cases:
        s = day_scores(th, aR, aT, c)
        z = aadc.idouble(0.0)
        for v in s:
            z = z + aadc.math.exp(v - s[0])                   # shift by the find-day score: stable
        loss = loss + aadc.math.log(z)                        # -log softmax(find) = log sum exp(s_j - s_find)
    for n in PARAMS:
        m, sd = PRIORS[n]
        loss = loss + 0.5 * ((th[n] - m) / sd) * ((th[n] - m) / sd)
    out = loss.mark_as_output()
    f.stop_recording()
    return f, handles, out


def record_scores(cases):
    """Tape: theta -> every day's score (for held-out evaluation)."""
    f = aadc.Functions()
    f.start_recording()
    th, handles = {}, []
    for n, v in zip(PARAMS, X0):
        th[n] = aadc.idouble(float(v))
        handles.append(th[n].mark_as_input())
    aR, aT = lag_weights(th["log_tau_r"]), lag_weights(th["log_tau_t"])
    outs = [[v.mark_as_output() for v in day_scores(th, aR, aT, c)] for c in cases]
    f.stop_recording()
    return f, handles, outs


def fit(cases, pool):
    f, h, out = record_loss(cases)

    def fun(x):
        vals, ders = aadc.evaluate(f, {out: h}, {hi: np.array([xi]) for hi, xi in zip(h, x)}, pool)
        return float(vals[out][0]), np.array([float(ders[out][hi][0]) for hi in h])

    r = minimize(fun, X0, jac=True, method="L-BFGS-B", bounds=BOUNDS)
    return r, fun


def scores_at(cases, x, pool):
    f, h, outs = record_scores(cases)
    vals, _ = aadc.evaluate(f, {o: [] for row in outs for o in row},
                            {hi: np.array([xi]) for hi, xi in zip(h, x)}, pool)
    return [[float(vals[o][0]) for o in row] for row in outs]


def auc_of(score_rows):
    pos = [r[0] for r in score_rows]
    neg = [v for r in score_rows for v in r[1:]]
    s = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return s / (len(pos) * len(neg))


def mean_loglik(score_rows):
    ll = 0.0
    for r in score_rows:
        m = max(r)
        ll += r[0] - m - math.log(sum(math.exp(v - m) for v in r))
    return ll / len(score_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default=",".join(DEFAULT_SPECIES))
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--write-fitted", default="",
                    help="comma-separated species whose all-years fit goes into fitted_params.json (the forecast)")
    ap.add_argument("--offline", action="store_true", help="use cached weather only; fetch nothing")
    ap.add_argument("--countries", default="PT", help="GBIF countries for fitting, e.g. PT,ES (test is always PT)")
    a = ap.parse_args()
    pool = aadc.ThreadPool(a.num_threads)
    QUOTA["offline"] = a.offline
    cache = load_cache()
    replay = {}
    rp = sorted(p for p in os.listdir(os.path.join(MUSHROOM_DIR, "runs")) if p.startswith("replay_gbif_"))
    if rp:
        with open(os.path.join(MUSHROOM_DIR, "runs", rp[-1])) as fh:
            for r in json.load(fh)["records"]:
                replay[(r["variant"], r["gbif"])] = [r["find_score"]] + r["control_scores"]

    report = {"model_version": MODEL_VERSION, "aadc": "classic 1.x (aadc-prod)", "generated": str(datetime.date.today()),
              "design": {"J": J, "ctrl": [CTRL_SPAN, CTRL_STEP, CTRL_GAP], "split_year": SPLIT_YEAR,
                         "priors": PRIORS}, "species": {}}
    for name in [s.strip() for s in a.species.split(",") if s.strip()]:
        cases = build_cases(name, cache, tuple(a.countries.split(",")))
        save_cache(cache)
        train = [c for c in cases if c["year"] <= SPLIT_YEAR]
        test = [c for c in cases if c["year"] > SPLIT_YEAR and c["country"] == "PT"]   # judged on Portugal only
        print(f"\n{name}: {len(cases)} cases ({len(train)} train <= {SPLIT_YEAR}, {len(test)} test)")
        res = {"cases": len(cases), "train": len(train), "test": len(test)}
        if len(train) >= 15 and len(test) >= 10:
            t0 = time.time()
            r, _ = fit(train, pool)
            sc = scores_at(test, r.x, pool)
            members = POOLS.get(name, [name])           # pooled: each find keeps its own species' rule score
            hand = [replay[(m, c["gbif"])] for c in test for m in members
                    if (m, c["gbif"]) in replay and len(replay[(m, c["gbif"])]) > 1]
            res["holdout"] = {
                "fit_seconds": round(time.time() - t0, 1), "converged": bool(r.success),
                "auc_fitted": auc_of(sc), "loglik_fitted": mean_loglik(sc),
                "loglik_chance": sum(-math.log(len(c["WE"])) for c in test) / len(test),
                "auc_hand_model": auc_of(hand) if hand else None, "hand_model_records": len(hand),
                "params_train": dict(zip(PARAMS, map(float, r.x)))}
            ho = res["holdout"]
            print(f"  held-out AUC fitted {ho['auc_fitted']:.3f} | hand-set model {ho['auc_hand_model']}"
                  f" | log-lik fitted {ho['loglik_fitted']:.3f} vs chance {ho['loglik_chance']:.3f}")
        if len(cases) >= 15:
            r, _ = fit(cases, pool)
            p = dict(zip(PARAMS, map(float, r.x)))
            res["params_all"] = p
            # Production score: today's s(t) WITHOUT the weekend nuisance term, expressed as a
            # percentile of s over every in-season control day in the data (all places, all years).
            x_prod = np.array(r.x)
            x_prod[PARAMS.index("w_we")] = 0.0
            ctrl = [v for row in scores_at(cases, x_prod, pool) for v in row[1:]]
            res["quantiles"] = [float(q) for q in np.quantile(ctrl, np.linspace(0, 1, 101))]
            res["readable"] = {"rain_window_days": math.exp(p["log_tau_r"]), "temp_window_days": math.exp(p["log_tau_t"]),
                               "T_opt_C": p["T_opt"], "T_sd_C": math.exp(p["log_sd"]), "w_rain": p["w_r"],
                               "w_temp": p["w_t"], "w_weekend": p["w_we"], "converged": bool(r.success)}
            print("  all years: " + ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
                                              for k, v in res["readable"].items()))
        report["species"][name] = res
    report["countries"] = a.countries
    adopt = [n.strip() for n in a.write_fitted.split(",") if n.strip()]
    if adopt:
        fitted = {}
        if os.path.exists(FITTED_FILE):
            with open(FITTED_FILE) as fh:
                fitted = json.load(fh)
        for n in adopt:
            r_ = report["species"][n]
            fitted[n] = {"params": r_["params_all"], "quantiles": r_["quantiles"], "J": J,
                         "holdout": r_.get("holdout"), "countries": a.countries,
                         "fitted": str(datetime.date.today()), "calibration_version": MODEL_VERSION}
        with open(FITTED_FILE, "w") as fh:
            json.dump(fitted, fh, indent=1)
        print(f"Wrote {FITTED_FILE}: {', '.join(adopt)}")
    out = os.path.join(MUSHROOM_DIR, "runs", f"calibration_{a.countries.replace(',', '-')}_{datetime.date.today()}.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nRecord: {out}")


if __name__ == "__main__":
    main()
