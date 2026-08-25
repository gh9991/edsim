# edsim — ED patient-flow simulation skeleton

Built for the **WA Health & Care Hackathon 2026** (WADSIH, 18–20 September).

The bet: ED flow has been a challenge in every edition (2023 ×2, 2024, 2025), the
last two grand prizes both went to *triage + bed management + simulation*, and
WADSIH has said 2026 leans harder into simulation and analytics. So the highest-
leverage prep is to arrive with a working simulator that only needs its data
layer swapped.

## The one design rule

Every data source is normalised to the **canonical schema** in `edsim/schema.py`.
The simulator, calibrator, metrics and any model you train read only from there.

```
portal.csv ─┐
mimic-iv-ed ─┼─► loader ──► canonical frame ──► calibrate ──► SimParams ──► simulate ──► metrics
synthea ────┘
```

On 15 September you write **one file** — `edsim/loaders/portal.py` — and nothing
downstream changes.

## Quick start

```bash
cd edsim && uv venv .venv && uv pip install -e ".[dev]"
.venv/bin/edsim demo
```

## What each command is for

| Command | Purpose |
|---|---|
| `edsim demo` | End-to-end smoke test, no downloads |
| `edsim units --lhn "East Metropolitan"` | List WA hospitals — EMHS is this year's partner |
| `edsim aihw H0632` | Calibrate volume + triage mix from **real** Royal Perth data |
| `edsim validate H0632` | Back-test: simulated vs AIHW's published on-time % |
| `edsim tune H0632 --out data/p.json` | Grid search the params AIHW can't tell us |
| `edsim simulate --params data/p.json --hourly` | Run it |
| `edsim sweep --ward-occupancy 0.55,0.75,0.85` | What-if grid — the pitch table |
| `edsim predict --explain 3` | Train + evaluate the admission model, explain one patient |
| `edsim inspect --path challenge.csv` | 15 Sept: see source columns vs canonical |
| `edsim calibrate --source portal --path challenge.csv` | 15 Sept: fit on real data |

## What the model actually says

The default sweep, on WA-calibrated demand:

```
 n_cubicles  ward_occupancy  on_time_pct  neat_pct  access_block_pct
         14            0.55         72.1      91.8               0.0
         14            0.85         28.8      65.0              37.6
         22            0.55         95.9      93.1               0.0
         22            0.85         49.4      68.0              48.7
```

Going from 14 to 22 cubicles when the ward is at 85% moves NEAT by 3 points.
**You cannot fix an emergency department by growing the emergency department.**
Access block is a ward-capacity problem wearing an ED costume. That single table
is the argument — it's modelled explicitly (`_patient` holds its cubicle while
waiting for a ward bed), not asserted.

## Calibrated on real WA data, not invented numbers

`edsim aihw H0632` pulls Royal Perth's actual figures from the open AIHW
MyHospitals API (no key required):

- 66,284 ED presentations in 2024–25 (181/day)
- ATS 1 treated on time: **100%** · ATS 3: **17%** · ATS 4: **31%**

## Reproducing a specific hospital

`edsim tune H0632` grid-searches the parameters AIHW cannot supply and gets
Royal Perth to **~8 pp mean absolute error** on published on-time performance:

| ATS | | AIHW observed | simulated |
|---|---|---|---|
| 1 | Resuscitation | 100% | 100% |
| 2 | Emergency | 60% | 64% |
| 3 | Urgent | 17% | 13% |
| 4 | Semi-urgent | 31% | 30% |
| 5 | Non-urgent | 65% | 32% |

ATS 5 is still off. It is 2% of presentations, so it carries almost no weight
in reality but full weight in the error metric — the honest read is that the
fast track's opening hours need calibrating, not that the model is broken.

### What the data actually said

The first hypothesis was that ATS 3 collapses because it is squeezed between
two protected streams. That was wrong, and the data says so plainly. Under a
single shared queue with median wait ~90 minutes, the ATS time targets alone
(30 / 60 / 120 min) produce roughly 21% / 37% / 60% on time — which is very
close to Royal Perth's actual 17% / 31% / 65%. **Most of the spread across
ATS 3-4-5 is an artefact of different targets applied to one common wait
distribution, not evidence of streaming.**

What genuinely needed modelling was the top of the scale:

- **Resus and fast track never board.** Admitted patients release those spaces
  immediately and wait for a ward bed in a corridor. Only main cubicles absorb
  access block. Without this, a gridlocked ED collapses ATS 1 along with
  everything else — which no real ED does.
- **ATS 1 preempts ATS 2 out of a resus bay.** ATS 2 is ~25% of arrivals; give
  it unpreemptable resus access and it saturates the bays, and ATS 1 on-time
  falls to single digits. With preemption, ATS 1 sits at exactly 100%.

Adding those two mechanisms took the error from **21.6 pp to 8.0 pp**. The
lesson worth carrying into the hackathon: the interesting number was never the
mean, it was *which category the model got wrong and in which direction*.

## Predicting admission — and why that is two different problems

The 2025 Department of Health challenge asked two things that sound alike:

> *"Which types of patients arriving at an ED will require hospital beds?"*
> *"Predict the **number** of inpatient admissions from ED triage."*

The first is a ranking problem — AUC is the right metric. The second is a
count forecast, where ranking is irrelevant and **calibration is everything**,
because expected bed demand is just the sum of the predicted probabilities. A
model can post a great AUC and still be useless to a bed manager. Most teams
will report AUC and stop. `edsim predict` reports both, and treats the daily
bed error as the headline:

```
                       auc: 0.708      <- which patients
                     brier: 0.189
          calibration_bias: 0.013      <- how many
            daily_mae_beds: 5.36       <- against 51 admissions/day
```

Also deliberate:

- **Leakage guard.** `features.LEAKY` names every column that only exists once
  the visit is over (`los_min`, `disposition`, `boarding_min`, …) and
  `assert_no_leakage` refuses to build a matrix containing one. The fastest way
  to a 0.99 AUC and a wrecked Q&A session is training on the outcome.
- **Temporal split, never random.** Train on the earliest slice, calibrate on
  the next, test on the latest. ED behaviour drifts; a random split lets the
  model see the future.
- **Crowding as a feature.** `ed_census_on_arrival` and
  `waiting_room_on_arrival` are reconstructed from earlier patients only. How
  busy the department already is, is knowable in real time and is what makes
  this an operational model rather than a clinical one.
- **Per-patient explanation.** `explain_patient` shifts each feature to the
  cohort median and reports what moves: *"ATS 4 rather than 3 — that alone
  takes this patient from 39% to 13%."* The 2024 Technical Achievement prize
  went to a team whose clinicians could drill into exactly that.

### How we know the pipeline works without real data

In the simulator, admission probability depends on **nothing but ATS**. That
puts a hard information ceiling on AUC, computable in closed form. The model
lands at **0.708 against a ceiling of 0.724** — near the bound, and a test
asserts it never exceeds it, because exceeding it would mean something leaked.
It also assigns ~zero importance to every other feature, correctly recovering
the generating mechanism rather than inventing signal.

That is the whole point of building this before the data arrives: on 15
September the loader changes and everything above runs unaltered.

## Deliberate modelling choices

- **Access block** — admitted patients keep their ED cubicle until a ward bed frees.
- **Three streams, not one queue** — ring-fenced resus bays (ATS 1–2, with
  ATS 1 preempting), a fast track open only part of the day (ATS 4–5), and
  main cubicles for everyone else plus overflow. ATS 1 bypasses triage.
- **Resus and fast track are decanted, never boarded** — see above.
- **Did-not-wait** — low-acuity patients abandon after a patience threshold.
  Waits are reported over patients *seen*, DNW separately — as EDs report it.
- **Non-homogeneous Poisson arrivals** via thinning, with a diurnal profile.
- **Provenance on every parameter.** `SimParams.notes` records what came from
  data and what is a default. `calibrate.check()` warns before you pitch a
  gridlocked configuration.

## Data sources, and the trap in each

| Source | Use it for | Do **not** |
|---|---|---|
| AIHW MyHospitals | WA volumes, triage mix, published performance | expect service times — it's aggregate |
| MIMIC-IV-ED demo (100 pts, open) | schema, pipeline, code that scales to the full set | — |
| MIMIC-IV-ED (425k stays, credentialed) | training a real admission/wait model | forget it has **no first-clinician-contact timestamp** |
| Synthea | FHIR structure, UI fixtures, load testing | fit queueing behaviour — it has no queue |
| Hackathon portal | the actual answer | commit it anywhere |

ATS (Australia, time-to-treatment) and ESI (US, resource-based) are both 1–5 and
mapped ordinally in `edsim/triage.py`. They are **not** clinically equivalent,
and they diverge most in the 3–4–5 band. Say so in the pitch.

## Layout

```
edsim/
  schema.py       canonical contract + validation
  triage.py       ATS/ESI, ACEM targets, NEAT
  aihw.py         AIHW MyHospitals API client (cached)
  calibrate.py    SimParams, fitting, capacity sizing, gridlock check, tuner
  sim.py          the SimPy model
  metrics.py      ATS on-time, NEAT, access block, DNW, census
  loaders/        mimic_demo · synthea · portal (fill in on the day)
tests/            17 tests, incl. the access-block behaviour above
```
