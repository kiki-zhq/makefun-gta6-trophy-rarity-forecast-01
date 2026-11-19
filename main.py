"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts each PS5 trophy's Exophase earn rate at the case target as five
quantiles (5/25/50/75/95%), scored by pinball loss on the logit scale.

Method — earners / players decomposition plus a damped-trend ensemble.

A trophy's rate is r = E / N, earners over players tracked.  Exophase does
not refresh continuously: the page sits unchanged for hours, then jumps when
it refreshes, and a refresh mostly moves N (new players appear in the
tracked count) while E moves only by what players actually earned.  So:

* collapse the history to its distinct reads (consecutive identical reads
  are one stale snapshot, not many observations);
* work in hours since launch, fitting the recent regime only — the last 24h
  of distinct snapshots.  Launch night is a different regime (tiny counts,
  steep rises as the first players reach each trophy), and a slope fitted
  over a longer window overstates the current slope for a saturating rate:
  the slope was much steeper early in the window than at the cut.  The
  house curve fits the last 24h for the same reason;
* estimate, per trophy, the recent per-hour slopes of log E_i, log N and
  logit(r_i), shrinking each slope toward its life-stage pool (trophies
  near the same rate level share a slope; a tier mixes saturated chapters
  with mid-lift-off grinds, and pooling across stages drags the saturated
  ones up the steep lift-off slope).  Logit slopes use only snapshots
  where the trophy has at least 2 earners: a 0.00% rate is censored, and
  its clamped logit would fake a vertical lift-off;
* extrapolate each slope with exponential damping: a slope s sustained for
  H hours moves the forecast by s * TAU * (1 - exp(-H / TAU)) with
  TAU = 48h.  A launch surge cannot sustain its lift-off slope for a week,
  and a rate cannot run away to 100% on a straight line.  Each candidate's
  total move from the last read is also capped at [-1.75, +1.5] logit;
* the median is the median of four logit forecasts: the E/N
  decomposition, the direct logit trend, a damped recent drift from
  the last two distinct snapshots (halved, capped at +/-1 logit), and
  persistence (the last read's rate).  Including persistence as a
  candidate keeps the ensemble honest when the trend candidates agree
  only because they share one window's slope: for a saturating rate the
  recent-drift and persistence reads are the ones that stay near the
  truth, and the median of four leans on them instead of the shared
  overshoot.

Flat history (a single distinct read, as on launch night) carries no trend,
so a documented refresh prior applies: the rules state the page sat
unchanged from 2026-11-18T23:50Z through 2026-11-19T04:00Z and refreshed at
04:30Z with 3,868 players tracked, up from 638 — a 6.06x jump with earners
nearly unchanged.  For a short horizon off a flat, sub-2000-player snapshot
we therefore expect one such refresh: N x6.06, earners x1.08 (a small hedge
for trophies earned in the meantime), every rate but a 100% one scaled down
by the same factor.  A trophy at 100% in every read stays 100%.

Spread is honest, not decorative: fit residuals plus slope uncertainty
grown over the damped extrapolation distance, plus a staleness term when
the last distinct read is older than the usual refresh gap, combined in
quadrature and clamped.  Short-horizon flat cases get s = 0.22 (the refresh
prior's own uncertainty); long horizons get up to 1.1 logit.

Python 3.13 standard library only, no network, no wall-clock input, no
randomness: two runs on the same case write the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
ORIGIN = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Documented first refresh of launch night (see module docstring).
FIRST_REFRESH_FACTOR = 3868.0 / 638.0  # 6.0627x players tracked
FLAT_EARNER_HEDGE = 1.08              # earners grow a little before refresh
FLAT_SPREAD = 0.22

TAU_HOURS = 48.0      # slope decay time for the damped extrapolation
MOVE_CAP_LO = -1.75   # widest logit move allowed from the last read
MOVE_CAP_HI = 1.5


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_origin(dt):
    return max(0.0, (dt - ORIGIN).total_seconds() / 3600.0)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 1.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (matches the starter's JS rounding)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def damped(slope, horizon_hours):
    """Integral of a slope decaying as exp(-t/TAU) over horizon_hours."""
    return slope * TAU_HOURS * (1.0 - math.exp(-horizon_hours / TAU_HOURS))


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_sd, se_b)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    xm = sum(w * x for w, x in zip(ws, xs)) / sw
    ym = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xm) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - xm) * (y - ym) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = ym - b * xm
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(1, len(xs) - 2)
    var = sum(w * r * r for w, r in zip(ws, resid)) / (sw * dof / max(1, len(xs)))
    resid_sd = math.sqrt(max(var, 0.0))
    se_b = resid_sd / math.sqrt(sxx) if sxx > 0 else 0.0
    return a, b, resid_sd, se_b


def median(values):
    v = sorted(values)
    n = len(v)
    if n == 0:
        return 0.0
    if n % 2:
        return v[n // 2]
    return 0.5 * (v[n // 2 - 1] + v[n // 2])


def distinct_reads(history):
    """Collapse consecutive identical snapshots; keep (slot, N, rates)."""
    out = []
    prev_key = None
    for h in history:
        rates = h.get("rates") or []
        key = (h.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        out.append((parse_time(h["slot"]), h.get("players_tracked"), rates))
    return out


def earners(rate, n):
    if rate is None or n is None:
        return None
    return rate * float(n) / 100.0


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    at_dt = parse_time(case["at"])

    reads = distinct_reads(history)
    last_slot, last_n, last_rates = reads[-1]
    n_reads = len(reads)

    # Fit window: only the recent regime extrapolates (see docstring).
    # Last 24h of distinct snapshots, widening only if that leaves
    # fewer than 3 points.  A longer window overstates the current slope
    # for a saturating rate: the slope inside it was much steeper early
    # in the window than it is at the cut.
    window = [rd for rd in reads
              if (at_dt - rd[0]).total_seconds() / 3600.0 <= 24.0]
    if len(window) < 3:
        window = reads
    fit_reads = window
    n_fit = len(fit_reads)

    h_last = hours_since_origin(last_slot)
    h_target = hours_since_origin(target)
    horizon_h = max(0.0, h_target - h_last)
    hs = [hours_since_origin(slot) for slot, _, _ in fit_reads]
    weights = [0.5 ** ((h_last - h) / 12.0) if n_fit >= 6 else 1.0
               for h in hs]

    # --- shared fits ---------------------------------------------------
    # log N against hours (one smooth series, large counts).
    n_slope = 0.0
    if n_fit >= 2:
        fit_n = wls(hs, [math.log(max(1.0, float(n)))
                         for _, n, _ in fit_reads], weights)
        if fit_n is not None:
            n_slope = fit_n[1]

    # Per-trophy slopes of log E and logit r against hours, with pools by
    # life stage — not by tier.  A tier mixes trophies at very different
    # stages (a saturated 54% chapter and a mid-lift-off 2% grind are both
    # bronze), and pooling across stages drags the saturated ones up the
    # steep lift-off slope.  Stage is the trophy's last rate: low (<2%),
    # mid (<20%), high (>=20%).
    def stage_of(rate):
        if rate is None:
            return "low"
        if rate < 2.0:
            return "low"
        if rate < 20.0:
            return "mid"
        return "high"

    stage_slopes_e = {}
    stage_slopes_r = {}
    per_trophy = []  # (fit_e, w_e, fit_r, w_r) per trophy index
    for i, tr in enumerate(trophies):
        hx_e, y_e, w_e = [], [], []
        hx_r, y_r, w_r = [], [], []
        for j, (slot, n, rates) in enumerate(fit_reads):
            r = rates[i] if i < len(rates) else None
            if r is None or n is None:
                continue
            e = earners(r, n)
            hx_e.append(hs[j])
            y_e.append(math.log(max(0.5, e) + 0.5))
            w_e.append(weights[j])
            # Logit slope only where the count is reliable: a 0.00% rate
            # is censored, and its clamped logit would fake a lift-off.
            if e is not None and e >= 2.0:
                hx_r.append(hs[j])
                y_r.append(logit(max(r, 0.02)))
                w_r.append(weights[j])
        fit_e = wls(hx_e, y_e, w_e) if len(hx_e) >= 2 else None
        fit_r = wls(hx_r, y_r, w_r) if len(hx_r) >= 2 else None
        per_trophy.append((fit_e, sum(w_e), fit_r, sum(w_r)))
        st = stage_of(last_rates[i] if i < len(last_rates) else None)
        if fit_e is not None:
            stage_slopes_e.setdefault(st, []).append(fit_e[1])
        if fit_r is not None:
            stage_slopes_r.setdefault(st, []).append(fit_r[1])
    pool_e = {t: median(v) for t, v in stage_slopes_e.items()}
    pool_r = {t: median(v) for t, v in stage_slopes_r.items()}
    global_e = median([b for v in stage_slopes_e.values() for b in v]) \
        if stage_slopes_e else 0.0
    global_r = median([b for v in stage_slopes_r.values() for b in v]) \
        if stage_slopes_r else 0.0

    # Median gap between distinct snapshots (the refresh cadence).
    gaps = [(reads[j + 1][0] - reads[j][0]).total_seconds() / 3600.0
            for j in range(n_reads - 1)]
    med_gap = median(gaps) if gaps else 0.0
    stale_hours = (at_dt - last_slot).total_seconds() / 3600.0

    rows = []
    for i, tr in enumerate(trophies):
        last_r = last_rates[i] if i < len(last_rates) else None
        if last_r is None:
            last_r = 0.0
        st = stage_of(last_r)
        lg_last = logit(last_r)

        # A trophy every tracked player holds, in every snapshot, stays 100%.
        if all((rates[i] if i < len(rates) else None) == 100
               for _, _, rates in reads):
            rows.append({"id": tr["id"], "q": [100, 100, 100, 100, 100]})
            continue

        # A trophy no tracked player has earned in any snapshot: no trend
        # exists to extrapolate, and predicting ~0% with a narrow spread
        # is crushed if even a few players reach it by the target (a
        # 0.00% rate at a 45k-player cut hides 0-2 earners).  Give it a
        # small day-7 prior by tier with a wide spread instead.
        if all((rates[i] if i < len(rates) else None) in (0.0, None)
               for _, _, rates in reads):
            tier = tr.get("tier", "?")
            med = 0.05 if tier in ("platinum", "gold") else 0.15
            rows.append({"id": tr["id"], "q": [
                round4(min(99.99, max(0.01,
                        sigmoid(logit(med) + z * 0.9) * 100.0)))
                for z in Z]})
            continue

        if n_reads == 1:
            # Flat history: no trend is visible; use the refresh prior for a
            # short horizon off a small launch-night snapshot, else persist.
            if case["horizon_minutes"] <= 240 and last_n and last_n <= 2000:
                med = last_r * FLAT_EARNER_HEDGE / FIRST_REFRESH_FACTOR
                spread = FLAT_SPREAD
            else:
                med = last_r
                spread = min(0.9, max(0.3, 0.5 * math.sqrt(horizon_days)))
            lg_med = logit(med)
        else:
            fit_e, w_e_sum, fit_r, w_r_sum = per_trophy[i]
            candidates = []
            spread_parts = [0.15]  # rounding + micro noise floor

            def cap_move(lg):
                return lg_last + max(MOVE_CAP_LO,
                                     min(MOVE_CAP_HI, lg - lg_last))

            # M1: earners / players decomposition, damped slopes.
            if fit_e is not None:
                b_e = (w_e_sum * fit_e[1] + 2.0 * pool_e.get(st, global_e)) \
                    / (w_e_sum + 2.0)
                move = damped(b_e, horizon_h) - damped(n_slope, horizon_h)
                candidates.append(cap_move(lg_last + move))
                spread_parts.append(fit_e[2])
                spread_parts.append(abs(fit_e[3])
                                    * TAU_HOURS
                                    * (1.0 - math.exp(-horizon_h / TAU_HOURS)))

            # M2: direct logit trend, damped slope, shrunk to stage pool.
            if fit_r is not None:
                b_r = (w_r_sum * fit_r[1] + 2.0 * pool_r.get(st, global_r)) \
                    / (w_r_sum + 2.0)
                move = damped(b_r, horizon_h)
                candidates.append(cap_move(lg_last + move))
                spread_parts.append(fit_r[2])
                spread_parts.append(abs(fit_r[3])
                                    * TAU_HOURS
                                    * (1.0 - math.exp(-horizon_h / TAU_HOURS)))

            # M3: damped recent drift from the last two distinct snapshots.
            if n_fit >= 2:
                r_prev = fit_reads[-2][2][i] \
                    if i < len(fit_reads[-2][2]) else None
                gap_h = hs[-1] - hs[-2]
                if r_prev is not None and gap_h > 1e-9:
                    slope = (logit(last_r) - logit(r_prev)) / gap_h
                    drift = 0.5 * damped(slope, horizon_h)
                    drift = max(-1.0, min(1.0, drift))
                    candidates.append(lg_last + drift)

            # Persistence as a fourth candidate (see docstring): it is the
            # vote that stays sane when the trend candidates share one
            # window's overshooting slope.
            candidates.append(lg_last)

            if not candidates:
                candidates.append(lg_last)
            lg_med = median(candidates)
            spread = math.sqrt(sum(p * p for p in spread_parts))
            spread += 0.10 * math.sqrt(horizon_days)
            if med_gap > 0 and stale_hours > med_gap:
                spread += 0.10
            spread = min(1.1, max(0.22, spread))

        qs = []
        for z in Z:
            q = sigmoid(lg_med + z * spread) * 100.0
            qs.append(round4(min(99.99, max(0.01, q))))
        # Keep quantiles non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tr["id"], "q": qs})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
