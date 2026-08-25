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
| `edsim information-value --path <mimic>` | What does an administrative extract give up? |
| `edsim inspect --path eddc.csv` | Source columns vs canonical, with a leakage verdict |
| `edsim qc --path eddc.csv` | Health-check an extract before modelling anything |
| `edsim calibrate --source portal --path eddc.csv` | Fit on real data |

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

## What an administrative extract gives up

Western Australia's Emergency Department Data Collection has four excellent
timestamps and **no clinical measurements at all** — no heart rate, no blood
pressure, no chief complaint. MIMIC-IV-ED is the mirror image: rich clinical
detail, no first-clinician-contact time. That makes MIMIC the one place to
measure what the administrative extract is missing, by handicapping a model
down to EDDC's information level and then giving the clinical fields back.

425,087 encounters · 205,504 patients · split **by patient**, never by row
(patients revisit 2.07 times, so a row split leaks the same person into both
sides). Stable across three seeds.

| Arm | AUC | vs A | cohort error | its own floor | excess |
|---|---|---|---|---|---|
| **A** · what EDDC records | 0.736 | — | 7.76% | 7.76% | **−0.00** |
| **B** · A + vitals | 0.772 | +0.036 | 7.69% | 7.50% | +0.19 |
| **C** · B + chief complaint | **0.829** | **+0.093** | 6.87% | 7.00% | −0.13 |

**All three arms sit exactly on their own statistical floor.** A bed count is
a random draw: even a perfectly calibrated model has an irreducible error of
`sqrt(2/π)·sqrt(Σp(1−p))/Σp`. Every arm hits it. So the apparent improvement
from 7.76% to 6.87% is **not better forecasting** — a sharper model pushes
probabilities away from 0.5, which mechanically shrinks the binomial variance
and lowers its own floor. Comparing each arm against *its own* floor separates
"forecasts better" from "is merely sharper", and the answer is that none of
them forecasts better, because none of them can.

That splits the question in two:

> **Counting beds** — the administrative extract is already at the limit.
> More clinical data cannot make the forecast better.
>
> **Flagging individual patients** — clinical data moves AUC from 0.736 to
> 0.829. Early warning at triage genuinely needs it.

And a third, more actionable result: **the free-text chief complaint (+0.057
over vitals) is worth more than the vitals themselves (+0.036).** If only one
field can be added to an extract, it is the one where the patient says what is
wrong.

Three limits, stated rather than buried: MIMIC is a US tertiary centre using
ESI rather than ATS, so this is an indicative magnitude, not a WA estimate.
MIMIC-IV-ED has no age, and no crowding is computable because every patient is
date-shifted into a different year — real EDDC has both, so arm A here is
*weaker* than a true EDDC model.

That last point makes the measured gap an upper bound, and the claim is
testable rather than merely plausible. Strengthening the baseline in steps and
re-measuring what vitals still add:

| Baseline | AUC | + vitals | marginal gain |
|---|---|---|---|
| ATS alone | 0.694 | 0.754 | **+0.060** |
| + sex, arrival mode | 0.731 | 0.769 | **+0.039** |
| + time features (= arm A) | 0.736 | 0.772 | **+0.036** |

The marginal value of vitals falls monotonically as the baseline improves, at a
strikingly steady exchange rate (0.58 then 0.56): **every 0.010 of baseline AUC
consumes about 0.0057 of what the vitals had left to give.** Sex and arrival
mode already carry part of the signal vitals encode — someone brought in by
ambulance is more likely to be sick — so the vitals arrive with less that is
new. Age and crowding would do the same, which is why the +0.093 measured here
is a ceiling. Two ratio points and a linear extrapolation make that directional,
not quantitative.

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

## The portal loader is already written

`loaders/portal.py` is built against the Department of Health's published field
definitions (*Linked Representative Synthetic EDDC / HMDC 2022 Data
Dictionary*, July 2025), not guessed from a sample — so the departure-status
codes, arrival modes and sex codes are the real ones, and `triage_category`
is read as the ATS it already is rather than converted from anything.

EDDC carries the two timestamps MIMIC does not have:

```
presentation_datetime                arrival
clinical_care_commencement_datetime  first seen  -> ATS on-time is measurable
bed_request_datetime                 bed asked for
discharge_datetime                   left        -> access block is measurable
```

Column names drift between extracts (`synth_person_ID` vs `person_ID`,
`metropolitan_hospital_flag` vs `metropolitan_flag`) and codes arrive either
numeric or pre-decoded, so every field goes through an alias list and a decoder
that accepts both. HMDC is joined with `merge_asof` on the episode that starts
*after* ED departure, within a bounded window — because the data dictionary
says inpatient admission time is when the patient leaves the ED, and a
same-person join would happily attach an unrelated admission from September.

### Leakage, taken from the definitions rather than guessed

| Field | Verdict |
|---|---|
| `mental_health_admission` | **leaky** — *"based on **departure status of admitted** and Principal Diagnoses"*. It contains the target |
| `primary_diagnosis_ICD10AM_chapter` | **leaky** — *"Chapter level roll of the Principal Diagnosis"*, coded after the episode ends |
| `bed_request_datetime` | **leaky** — requesting a bed *is* the admission decision |
| `mental_health_attendance` | suspect — partly derived from the post-hoc diagnosis |
| `potentially_avoidable_..._attendance` | suspect — "pre-calculated" from undisclosed variables |

`edsim inspect` prints this verdict beside every source column.

## Check the data before you model it

Synthetic data is generated, and generators leave fingerprints. `edsim qc` runs
the checks that have actually caught something:

```
FAIL  dates not shifted        11.4/day over 37,339 days (102.2 yr)
      ← span 102 years is far longer than any real extract.
        Crowding features and daily forecasts are invalid
```

Others catch waiting time that does not vary with triage category, admission
that is a deterministic function of triage rather than a probability, an ATS 1
median wait in the tens of minutes, out-of-order timestamps, and an implausible
acuity mix. A check that cannot run says so rather than silently passing.

A failure here is a finding, not a blocker. The Department's own lead data
scientist builds this data; "your generator makes waiting time independent of
triage category" is worth more than another dashboard.

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
