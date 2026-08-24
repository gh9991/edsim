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

## Known open problem — this is your first real task

`edsim tune H0632` currently bottoms out around **21 pp mean absolute error**.
The model cannot simultaneously reproduce ATS 1 at 100% and ATS 3 at 17%: a
single priority queue that starves ATS 3 that badly also starves ATS 1.

Royal Perth's real pattern implies something this model doesn't have yet —
likely streaming (fast-track/ambulatory running as a separate resource pool),
or time-varying staffing, or a nurse resource distinct from the physical
cubicle. Fixing that is worth more than any amount of UI polish, and it is
exactly what a DoH judge will probe.

## Deliberate modelling choices

- **Access block** — admitted patients keep their ED cubicle until a ward bed frees.
- **Reserved resus bays** — ATS 1–2 get a protected pool; ATS 1 bypasses triage.
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
