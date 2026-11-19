# GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kiki-zhq/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kiki-zhq/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each Grand Theft Auto VI PlayStation 5 trophy's Exophase earn rate one week after launch, as five quantiles per trophy (5/25/50/75/95%), for make.fun's `gta6-trophy-rarity-forecast-01` bounty ($250 in USDC).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only) reads every `<id>.in` case in the input directory and writes `<id>.out` per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Method

A trophy's rate is earners over players tracked, `r = E / N`, and Exophase refreshes in jumps: the page sits unchanged for hours, then a refresh mostly moves `N` (new players appear in the tracked count) while `E` moves only by what players actually earned. The forecaster therefore:

- collapses the history to its distinct reads (consecutive identical reads are one stale snapshot, not many observations);
- fits the recent regime only — the last 24h of distinct snapshots — estimating per-hour slopes of log `E`, log `N` and logit(`r`), shrunk toward the pool of trophies at the same life stage (rate level), since a tier mixes saturated chapters with mid-lift-off grinds;
- extrapolates each slope with exponential damping (48h decay time) and caps each candidate's total logit move from the last read, so a lift-off slope cannot run a rate to 100%;
- takes the median of four logit forecasts: the `E/N` decomposition, the direct logit trend, a damped recent drift, and persistence. Persistence is the vote that stays sane when the trend candidates share one window's overshooting slope.

Flat history (a single distinct read, as on launch night) carries no trend, so a documented refresh prior applies: the rules state the page sat unchanged through 2026-11-19T04:00Z and refreshed at 04:30Z with 3,868 players tracked, up from 638 — a 6.06x jump with earners nearly unchanged. For a short horizon off a flat, sub-2000-player snapshot the forecast is one such refresh (N x6.06, earners x1.08). A trophy at 100% in every read stays 100%; a trophy no one has earned gets a small tier prior with a wide spread instead of ~0%.

Spread is honest: fit residuals plus slope uncertainty over the damped extrapolation distance, plus a staleness term, combined in quadrature.

## Validation

Local checks against the kit's practice cases and synthetic final-like cases (cohort/lag dynamics with refresh dilution, a saturating smooth world, and a continued-surge world), scored with the contest's own pinball loss on the logit scale:

| case | this entry | persistence baseline | house curve baseline |
|---|---|---|---|
| practice (refresh-dilution truth) | 0.05–0.06 | 1.52–1.69 | 1.51 |
| synthetic lag world | 0.25 | 0.82 | 1.66 |
| synthetic smooth world | 0.32 | 0.53 | 1.18 |
| synthetic surge world | 0.72 | 1.67 | 4.33 |

Practice loss is computed against the dilution of the launch-night snapshot (earners constant, players 638 → 3,868, the 100% trophy unchanged), with the earners-growth factor varied 1.00–1.20. Runs are deterministic and take ~0.1s per case.

---

<sub>Entered on [make.fun](https://make.fun/?ref=t49c36pe&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
