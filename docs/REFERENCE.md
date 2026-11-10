<div align="center">

# 🎯 cs2edge

**A calibrated tipster for professional CS2.**

It estimates the win probability before the match, *chooses which matches it
speaks on*, explains why, archives its calls — and publishes its own track record.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![Engine](https://img.shields.io/badge/engine-NumPy%20only-013243?logo=numpy&logoColor=white)
[![Tests](https://github.com/Bdexez/Cs2edge/actions/workflows/tests.yml/badge.svg)](https://github.com/Bdexez/Cs2edge/actions/workflows/tests.yml)
![Probabilities](https://img.shields.io/badge/probabilities-calibrated-1f6feb)

</div>

---

> [!TIP]
> **Two rules guided the whole build.**
> 1. **Calibrated probabilities, not blind tips.** A calibrated model should win roughly 70 times out of 100 when it announces 70 %. The backtest measures that relationship on the supplied data.
> 2. **Silence beats filler.** On the shipped data, it discards 17 % of matches. Selectivity *is* the product: a tipster who has an opinion on everything gives you no way to tell convictions from guesses.

### In short

- 🃏 **A slate of tips** ranked by conviction, each line backed by numbers.
- 🗳️ **The veto structure** unrolled map by map — where naive models get it most wrong.
- 💹 **Two modes**: "who wins" without odds, "where the market is wrong" with odds.
- 📊 **An honest track record**: announced *versus* observed, sampling margin included.
- 🧮 **NumPy-only core**: logistic regression, calibration and pipeline written by hand.

---

## Contents

- ⚡ [Quick start](#quick-start)
- 🃏 [The tipster](#the-tipster)
- 🗳️ [The veto, and why it changes everything](#the-veto-and-why-it-changes-everything)
- 🔎 [Analysing a single match](#analysing-a-single-match)
- 📊 [Being accountable](#being-accountable)
- ⚙️ [How it works](#how-it-works)
- 📥 [Plugging in your own data](#plugging-in-your-own-data)
- 🔬 [Measuring, and distrusting flattering numbers](#measuring-and-distrusting-flattering-numbers)
- 🧪 [What measurement corrected](#what-measurement-corrected)
- ⚠️ [What it does not do](#what-it-does-not-do)
- 🗂️ [Code structure](#code-structure)

---

## Quick start

```bash
git clone https://github.com/Bdexez/Cs2edge.git
cd Cs2edge
python -m venv .venv && .venv/bin/pip install -e .

# optional: adds a GBDT to the ensemble (+ pandas export)
.venv/bin/pip install -e ".[gbdt]"
```

The core depends only on **NumPy** — the logistic regression, the calibration and
the whole pipeline are written by hand. `scikit-learn` is optional.

```bash
cs2edge sample                              # synthetic history to try it out
cs2edge fixtures --out data/fixtures.csv    # upcoming matches (example)
cs2edge slate --fixtures data/fixtures.csv  # 🃏 the slate of tips
cs2edge tipster-backtest                    # 📊 the tipster's track record
cs2edge backtest --by-year                  # 📊 the model's track record
cs2edge ratings --top 15                    # Elo ranking
cs2edge predict navi g2 --bo 3 --odds-a 1.75 --odds-b 2.10
python -m tests.test_smoke                  # 32 end-to-end tests
```

---

## The tipster

```
$ cs2edge slate --fixtures data/fixtures.csv --record data/journal.csv

PRONOSTICS (8 matchs)
========================================================================

falcons vs heroic   bo3, LAN   [Exemple]
  2026-09-09
  -> vainqueur : falcons   92%   [solide]   cote juste 1.09
       . Ecart de niveau : 219 points Elo en faveur de falcons (1807 contre 1588).
       . Le niveau des adversaires affrontes penche vers heroic.
       . Terrain projete apres veto (lecture brute) : anubis (pick falcons, 92%),
         nuke (pick heroic, 90%), dust2 (decisive, 90%).
       . Pool desequilibre : 84% de chances que la serie se termine sans
         derniere map.
  -> total maps -2.5 : oui   84%   [solide]   cote juste 1.19
       . Loi du score issue du veto projete : 2-0 81%, 2-1 11%, 1-2 5%, 0-2 3%.
  -> handicap -1.5 : falcons   81%   [solide]   cote juste 1.24
       . Pool desequilibre en faveur de falcons : le veto lui laisse ses
         meilleurs terrains, ce qui rend le sweep (2-0) plus probable (81%).
       . Pronostic correle au vainqueur : ne pas cumuler les mises.

[...]

ECARTES (2)
========================================================================
  falcons vs 9z            match trop serre (50%) pour un pronostic assume
  9z vs faze               match trop serre (58%) pour un pronostic assume
```

The slate is **ranked by conviction**. Every tip carries a tier, a fair price, and
the quantified reasons that produced it — including the ones that argue
**against** the tip, because reasoning that only mentions what suits it is not
reasoning.

### Two modes, not to be confused

|  | 🎯 without odds | 💹 with odds |
|---|---|---|
| **question asked** | "who wins?" | "where is the market wrong?" |
| **criterion** | probability ≥ 60 % | edge ≥ 4 pts after removing the margin |
| **what is optimised** | hit rate | expected value |
| **hit rate** | high | **structurally lower** |

> [!IMPORTANT]
> This distinction is the blind spot of most tipsters. In market mode, value is
> found mostly on underdogs: you lose more often, and that is normal. Advertising
> your hit rate to sell odds-based tips is selling an illusion.

Filling in **a single** odds column of a fixtures file automatically switches to
market mode — each market is evaluated independently.

**In market mode, two safeguards:**

- 🛡️ **No negative-EV bet.** The edge is measured against the price *stripped of
  its margin*; the expected value, however, against the price *actually offered*.
  In a high-margin market, beating the bookmaker's "true" probability is not
  enough to beat its price — so the tipster never announces a bet whose EV is
  negative.
- 🎚️ **Cumulative stake capped per match.** Winner, total maps and handicap all
  draw on the same outcome: adding them at full Kelly stake would mean betting two
  to three times the intended size on a single position. Their sum is brought back
  to the cap of a single bet.

---

## The veto, and why it changes everything

A bo3 is not played on three randomly drawn maps: map 1 is picked by A, map 2 by
B, and the decider is whatever survives the bans. `cs2edge` unrolls that sequence
and derives the **full score distribution** from it.

On the winner, the effect is negligible. On derived markets, it is massive —
measured on a polarised pool:

| | i.i.d. model | with the veto structure |
|---|---:|---:|
| winner | 50.0 % | 50.0 % |
| 2-0 win | 25.0 % | **12.8 %** |
| 3-map series | 50.0 % | **74.5 %** |

That is exactly where the "total maps", "handicap" and "correct score" markets
live — so exactly where the approximation is expensive. Those prices are often set
from a model with no veto.

> [!NOTE]
> The **sweep handicap** (-1.5 in a bo3, -2.5 in a bo5) falls straight out of
> that structure. A bo5 -1.5 also covers 3-1. An unbalanced pool
> makes the sweep far more likely than an i.i.d. model believes. The tipster
> raises it to its confidence tier — validated down into the 0.60–0.70 band on the
> backtest — and flags that it is **correlated** with the winner: you do not stack
> two full stakes on one and the same conviction.

### Map-to-map carry-over (momentum)

Does winning the previous map change the next one, beyond what the model already
knows about the two teams? `slate` **measures** it on the history and only applies
it if it exceeds the noise (`--momentum auto`, the default). It is the only
quantity left at zero by default, precisely because it is easy to hallucinate:
many public models impose it by intuition, which amounts to inventing signal.

The coefficient applies to the cumulative map-win balance before the next map,
matching the series simulator. It is estimated using historical predictions
whose training ends before their target day, with standard errors grouped by
match. The diagnostic measures residual association; it does not identify a
causal effect of winning a map.

---

## Analysing a single match

```
$ cs2edge predict 9z faze --bo 3 --lan

9z  vs  faze   [bo3, LAN, 2026-09-06]

  9z                58.6%   cote juste 1.71
  faze              41.4%   cote juste 2.41

  Favori : 9z (58.6%) - confiance correcte
  Modele match 52.4% | reconstitue par maps 64.8% (vu de 9z)

  Ce qui pese (contribution au logit, vu de 9z) :
    round_form_diff        -1.006  ############
    perf_residual_diff     +0.622  #######
    pool_mean_diff         +0.179  ##
    elo_diff               +0.160  #
    form_diff              +0.125  #
    trajectory_diff        -0.109  #

  Pool de maps (avantage en points Elo, vu de 9z) :
    dust2       +280.7   p=71.7%   [55v30 maps]
    train       +234.1   p=66.6%   [56v60 maps]
    ancient      +84.3   p=59.6%   [54v51 maps]
    anubis       +48.5   p=61.1%   [53v45 maps]
    mirage       +22.8   p=54.2%   [30v57 maps]
    nuke         +21.5   p=53.7%   [47v59 maps]
    inferno      -57.9   p=47.7%   [51v49 maps]

  Historique : 9z 346 maps (derniere 2026-09-01), faze 351 maps
               (derniere 2026-09-06), confrontations directes 12
```

<sub>*(real output, on the synthetic data shipped with the project)*</sub>

Given odds, it says whether there is an exploitable edge **after removing the
bookmaker's margin** — and refuses to conclude when the edge is too good to be
true.

---

## Being accountable

A tipster who does not archive their calls cannot be evaluated — and without a
timestamped record, nothing stops anyone from mostly remembering the wins.

```bash
cs2edge slate --fixtures f.csv --record data/journal.csv   # archive BEFORE the match
cs2edge settle --journal data/journal.csv                  # settle AFTERWARDS
cs2edge journal --journal data/journal.csv                 # the track record
```

The journal is **append-only**: settling completes the rows, it never rewrites
them. Both operations are idempotent.

### The tipster's track record

`cs2edge tipster-backtest` replays the whole strategy over the history, using at
each instant only the past:

```
Bilan du pronostiqueur : 1376 matchs presentes, 235 ecartes (17%), 2147 pronostics emis.

Par palier de confiance (un palier qui ne tient pas sa promesse ment) :
  palier             n   annonce   observe    ecart    marge
  solide          1653    86.0%     87.6%    +1.6%  +/-1.7%
  interessant      494    64.8%     63.8%    -1.1%  +/-4.3%

Par marche :
  marche                   n   annonce   observe    ecart    marge
  handicap -1.5          405    79.3%     82.0%    +2.7%  +/-4.0%
  total maps -2.5        523    77.6%     80.1%    +2.5%  +/-3.6%
  total maps -4.5         56    86.1%     85.7%    -0.4%  +/-9.2%
  vainqueur             1141    83.3%     83.1%    -0.2%  +/-2.2%
```

**A tier has to keep its promise.** The `ecart` column compares the announced with
the realised. A tier that announces 70 % and realises 60 % lies to its reader,
even if it wins more often than it loses.

**A gap is only a defect if it exceeds the noise.** The `marge` column gives two
standard errors. Below that, no conclusion — that is the classic failing of
tipster track records, which comment on twenty rows as if they demonstrated
something.

> This backtest shaped the product, not the other way round: it had a tier
> removed, a calibration corrected, and derived markets opened only where they keep
> their promise.

---

## How it works

**1. Elo, global and per map.** One global rating per team, plus one rating per
(team, map) pair. Three refinements over a textbook Elo: damped margin of victory
(crushing an underdog earns little), K modulated by the event tier, and regression
towards the mean after inactivity or a roster change.

**2. Features, in a single chronological pass.** Each row is computed with the
state of the teams *before* the match, then the state is updated. The same object
serves training and production prediction — which eliminates train/serve
divergence. On the menu: Elo gap, pool advantage, recency-weighted form, round
differential, head-to-head history, rest days, recent workload, strength of
schedule, current streak, roster tenure.

**3. Two independent estimators.**
- A **match model**: ridge logistic regression without an intercept, solved by
  IRLS, plus an optional GBDT.
- A **map-level reading**: probability of winning each of the 7 active-duty maps,
  from which the best and the worst are removed (each side bans what suits it),
  then converted into a series.

They are blended in equal parts. That weight is not chosen by intuition: it is
measured by backtest, and `cs2edge backtest` displays the optimum on *your* data.

<details>
<summary><b>4. Two structural safeguards</b> — antisymmetry & decorrelation</summary>

<br>

*Antisymmetry.* All features are differences (A − B), the model has no intercept,
and it is trained on the mirrored data `(X, y)` and `(−X, 1−y)`. Result:
`p(A beats B) + p(B beats A) = 1` to machine precision. Without that, the order in
which you enter the teams shifts the prediction — the classic silent bug of this
kind of system.

*Decorrelation.* Form, pool and schedule are near-duplicates of the Elo gap.
Without removal, the ridge penalty spreads that shared signal at random across the
columns and `elo_diff` can come out with a *negative* weight. The model therefore
projects each column out of the Elo gap — a linear transformation, which does not
change what the model can represent, only the readability of the coefficients.

</details>

**5. From map level to series level.** The veto is unrolled (each side bans its
worst map, picks its best), which gives the maps actually played and their order.
The full score distribution follows by enumeration.

**6. Two calibrated marginals, reconciled.** The winner comes from the match model;
the series length is calibrated separately on the history. An iterative
proportional fitting imposes both margins on the score distribution without
breaking its internal structure. Without that step, the "winner" tip and the
"correct score" tip could contradict each other — a defect immediately visible to
an attentive reader.

**7. Temperature calibration.** A single parameter, fitted on a temporal holdout.
It preserves ordering and symmetry, and corrects systematic over-confidence. A
badly calibrated probability is worse than useless as soon as you compare it to a
price.

---

## Plugging in your own data

The project scrapes nothing: it reads a **map-level** CSV, which makes it agnostic
about the source (HLTV export, bo3.gg, Liquipedia, a Kaggle dataset, an in-house
database).

```csv
match_id,date,team_a,team_b,map,rounds_a,rounds_b,best_of,event,tier,lan,map_index
2380145,2026-08-14,Vitality,Spirit,inferno,13,9,3,IEM Cologne,s,1,0
2380145,2026-08-14,Vitality,Spirit,nuke,10,13,3,IEM Cologne,s,1,1
```

| column | required | role |
|---|:---:|---|
| `match_id` | ✅ | groups the maps of one series |
| `date`, `team_a`, `team_b`, `map`, `rounds_a`, `rounds_b` | ✅ | — |
| `best_of` | — | defaults to 3 |
| `tier` | — | `s/a/b/c/d` — weights the result's influence on the Elo |
| `lan` | — | LAN vs online |
| `map_index` | — | order of the maps within the series |
| `roster_a`, `roster_b` | — | comma-separated handles — detects roster changes |

Dates accept `YYYY-MM-DD`, `DD/MM/YYYY`, `YYYY/MM/DD` and `DD-MM-YYYY`.
ISO timestamps with seconds, optional microseconds and an optional UTC offset
or `Z` are also accepted, with `T` or a space between date and time. The written
calendar day is preserved; timestamps are not converted to UTC. The entire value
must be valid: suffixes and impossible times are rejected.

History, fixture and journal CSV readers reject duplicate headers, unterminated
quotes and rows with too few or too many cells. Leave optional cells explicitly
empty, including trailing cells declared in the header. UTF-8 BOMs, quoted commas
and quoted multiline fields are supported. Errors identify the file and physical
line when applicable. An existing journal with a damaged structure is refused
before appending or settling; older journal headers remain supported.

### Upcoming matches

```csv
date,team_a,team_b,best_of,lan,event,odds_a,odds_b,odds_over,odds_under,odds_a_sweep,odds_b_sweep
2026-09-14,Vitality,Spirit,3,1,IEM Cologne,1.75,2.10,1.95,1.85,3.40,6.50
```

Only `date`, `team_a`, `team_b` are required. Odds are optional and switch it from
analysis mode to market mode. `odds_a_sweep` / `odds_b_sweep` are the **map
handicap** prices (the team wins without dropping a map): that is the market where
the model's edge is largest, because books often open it from an i.i.d. model
blind to the veto. Lacking the complementary `+1.5` price to strip the margin, the
edge there is measured against the raw price — a deliberately conservative reading.

Team names are normalised automatically (`Natus Vincere` → `navi`, `de_Dust2` →
`dust2`). From code, `sources.from_records()` accepts dicts coming from an API
directly.

> [!TIP]
> `roster_a`/`roster_b` are missing from most public exports, and that is a shame:
> a two-player change invalidates a good part of a team's history, and the model
> knows how to account for it when you give it the information.

---

## Reusing a trained analyzer

For Python callers, `Analyzer.load_maps()` and `Analyzer.train()` replace state
only after their work succeeds. A failed import or training attempt preserves
the previous models, history and configuration. A successful import clears the
models and requires training again. If automatic momentum was enabled, its old
estimate is cleared on import and recalculated during training using the new
data and options. Explicitly configured momentum is preserved.

```bash
cs2edge train --no-gbdt --bundle models/analyzer.cs2edge
cs2edge predict navi g2 --model models/analyzer.cs2edge
cs2edge slate --model models/analyzer.cs2edge --fixtures data/fixtures.csv
```

The single archive includes the history, Elo configuration, feature schemas and
both fitted predictors. Future predictions need no source CSV and fit no new
model. Historical dates still rebuild using only earlier days. Data and training
options (`--data`, `--warmup`, `--no-gbdt`, `--elo-config`) conflict with `--model`;
train a new bundle to change them. Explicit `--momentum 0` on a slate overrides
the saved value and its automatic estimation mode.

`Analyzer.save(path)` and `Analyzer.load(path)` expose the same feature in Python.
Loading checks component hashes, schemas and model/history compatibility. GBDT
bundles require the same scikit-learn version and use pickle: load bundles only
from trusted sources. `train --out models/match.json` retains the match-only
JSON export and does not produce a complete analyzer.

New v2 archives also store the predictions used for calibration, with their
training cutoff dates. This lets a future slate use the default momentum and
length calibration without refitting classifiers. V1 archives remain readable;
their calibration history is built lazily if requested. Retraining or loading
different data invalidates the cache.

Probability and policy inputs must be finite and within their domains. In
particular, abstention must be between 0.5 and 1 and calibration temperature
must be positive. The implemented bo3/bo5 veto requires seven distinct maps;
a series distribution requires exactly `best_of` ordered, distinct maps.

## Measuring, and distrusting flattering numbers

```bash
cs2edge backtest --by-year
```

The backtest is **walk-forward**: train on the past only, predict the next block,
retrain, move on. It is the only honest protocol here. A random cross-validation
would give markedly better scores and entirely false ones, because a given match's
Elo rating already encodes later matches.

The report shows the **log-loss** and a `skill score` (log-loss reduction against a
coin flip), compared against three references: match model alone, map-level
reading alone, and raw Elo.

The gain over Elo includes a paired 95% bootstrap interval. Entire calendar
blocks are resampled for both models together: `--bootstrap-samples` defaults
to 1000 (`0` disables it), and `--block-days` defaults to 7. An interval crossing
zero leaves the advantage uncertain. The interval is conditional on the tested
period; use larger blocks to check sensitivity to longer dependencies.

> [!WARNING]
> **Accuracy is a poor indicator.** Raw Elo often shows *higher* accuracy than the
> full model while having a markedly worse log-loss: it picks the winner slightly
> better, but with badly calibrated probabilities — so unusable against a market.
> **Log-loss** and **ECE** are what count.

The report also shows a reliability curve: for each band of announced probability,
the frequency actually observed. A systematic gap of more than 5 points in a
well-populated band signals a calibration problem.

---

## What measurement corrected

These decisions do not come from an intuition about CS, but from a number saying
something was wrong.

**🗑️ A tier was removed.** A third "tight" level covered the 57–60 % band. Over the
history: 62 tips, 58.6 % announced, 66.1 % observed. A three-point band, a tiny
sample, a gap entirely compatible with noise — chance presented as judgement. Two
tiers that hold beat three of which one lies.

**🔧 A parameter was tried, then dropped.** The "total maps" market announced 76.9 %
and realised 87.4 %. First attempt: a "dispersion" parameter. Unstable from fold to
fold (0.10 to 1.05) without correcting the measured defect. The real culprit was
elsewhere: the score distribution was anchored on a deliberately conservative winner
probability, which pulled the whole distribution towards the balance point. The right
answer was not to distort the veto, but to calibrate the second marginal as the first
one already was. After correction: **+2.0 % gap instead of +10.5 %**.

**➕ Derived markets were opened more widely.** Total maps and handicap were reserved
for the solid tier (≥ 0.70). Measured over twelve simulated worlds, the 0.60–0.70 band
keeps its promise (total −0.5 %, handicap +1.3 %): issuing them from the abstention
threshold onwards **cleaned up the "interesting" tier** (from −5.1 % to −1.1 %) and
increased coverage by **~40 %**, without any market stepping outside its margin.

---

## What it does not do

> [!WARNING]
> **The shipped data is synthetic.** `cs2edge sample` simulates a credible history,
> which makes it possible to run the whole chain without real data — but a simulated
> world does not establish performance on real professional matches. The current
> reproducible results are in the [latest verification report](NOUVEAUX_AXES_IMPLEMENTES.md).
> Real-world performance requires a representative chronological dataset and
> independent evaluation; no accuracy range or betting return is established here.

**The ceiling is low, and it is structural.** Pro CS2 is a high-variance sport: a map
is played over 24 rounds, a bo3 turns on two duels. An irreducible part of the outcome
is not predictable from history, whatever the model.

**The model ignores anything that is not in past results** — a last-minute stand-in, a
sick player, a visa problem, a transfer announced the day before, nothing at stake, jet
lag, a hardware failure. That is why `predict` caps confidence at 96 %, shrinks
probabilities when the history is thin or stale, and warns you in case of a massive
disagreement with the market — in that case, it is almost always the market that knows
something.

**A high hit rate proves nothing.** Announcing that the world number one beats number
thirty, and being right 90 % of the time, requires no model. The track record therefore
shows *announced versus observed* rather than a raw rate: the only interesting question
is whether the announced number was right.

**The market is a serious opponent.** The closing prices of the big bookmakers are, over
the long run, better calibrated than what this kind of model produces. The functions in
`odds.py` (power-method margin removal, expected value, fractional Kelly capped at a
quarter) make the comparison honest — they do not promise a return.

---

## Code structure

```
cs2edge/
  schema.py     data model, name normalisation, CSV reading
  sources.py    data inputs + history simulator
  ratings.py    global and per-map Elo (margin, tier, inactivity, roster)
  features.py   leak-free extraction, dataset construction
  model.py      IRLS logistic regression, decorrelation, calibration, ensemble
  metrics.py    log-loss, Brier, ECE, reliability curve
  backtest.py   walk-forward, comparison of the estimators
  odds.py       margin removal, expected value, Kelly
  markets.py    veto, score distribution, derived markets, margin calibration
  predict.py    orchestration and prediction report
  snapshot.py   complete analyzer archive, schema and integrity checks
  pronostic.py  selectivity, tiers, reasoning, tipster backtest
  journal.py    archiving of calls, settling, track record
  cli.py        command-line interface
tests/          unittest suite, including 32 historical scenarios and optional GBDT tests
```

The tipster's decision rules live in a **pure function**, called both by the live
service and by its own backtest. Decision logic written in two copies always ends up
diverging, and the published track record becomes false without anyone noticing.

---

<div align="center">

### On usage

</div>

This tool is a **measuring instrument**: it quantifies what the history allows one to
say, and it is explicit about what it ignores. Its tips are worth something for
analysing a match — they constitute neither advice nor a guarantee, and a calibrated
model does not remove variance: it quantifies it.

> A 70 % tip is wrong three times out of ten. That is the expected behaviour, not an
> anomaly — and it is also why a tipster is only judged over hundreds of calls, never
> over the last one.
