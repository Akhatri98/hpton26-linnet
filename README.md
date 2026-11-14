# LINNET

**Medication timing, checked against the label.** Given a patient's medications and their day (meals, sleep), LINNET finds a dose schedule that obeys every timing rule on the FDA labels, or proves none exists and shows exactly which label sentences conflict.

Built solo at HackPrinceton Fall 2026 (Healthcare track).

---

## The problem

Drug labels carry timing rules: *take on an empty stomach*, *every 4 hours*, *at least 2 hours before or 6 hours after antacids*. A patient on four or five medications can have a dozen of these interacting, and pharmacists work them out by hand. Published scheduling methods place one dose at a time and give up when they get stuck; when they give up on a regimen that *was* schedulable, that is a false alarm, and the patient may be told to change a prescription they didn't need to.

## What LINNET answers

| Tier | Meaning | What the pharmacist sees |
|---|---|---|
| **0** | Fits the patient's day as it is | A medication card and a 24-hour timeline |
| **1** | Fits with the smallest change | Ranked options: move lunch 30 min, wake 15 min earlier, drop one as-needed dose |
| **2** | The prescriptions alone cannot be scheduled, even under the loosest reading of every label | A minimal conflict: the prescriptions and the exact label sentences that clash, with links to DailyMed |

Example tier 2, found in real labels: nimodipine *every 4 hours* (6 doses a day) together with colestipol's *other drugs at least one hour before or four hours after*. No 24-hour day fits both. LINNET names the two sentences.

## How it works

```
FDA label (public)                                   Patient (stays in memory)
     |                                                        |
     v                                                        v
[CODE]  number the sentences                       [CODE]  build the regimen
[MODEL] propose rules, citing sentence ids                    |
[CODE]  check: sentence exists, every number                  v
        is on the label, schema, scope             [CODE]  Z3 solver: tiers 0 / 1 / 2
[HUMAN] pharmacist signs off each rule  -------->  [CODE]  exact re-check of every schedule
                                                   [CODE]  shrink conflicts to minimal
```

### What is algorithmic, and what is a language model

| Step | Kind | Notes |
|---|---|---|
| Fetching labels, splitting sentences | deterministic | openFDA / DailyMed |
| Proposing rules from label text | **language model** | the only probabilistic step; cites sentence ids, never quotes (the quote is copied by code) |
| Checking proposals | deterministic | sentence exists; every number appears in that sentence; schema; "every other drug" only if the sentence says so; group members must be named (or be a salt of a named element); a strength must be one the label lists and a maximum must name one of its ingredients; rejects a per-kg dose read as flat, a maximum or titration step read as a dose, "every N hours" read as a spacing rule (the prescription sets the regimen), and duplicates |
| Strengths and ingredients on a label | deterministic | read by code from How Supplied, Description and openFDA; the review card shows any strength with no regimen in red |
| Review queue | deterministic | label sentences with numbers that no accepted row uses, sorted so likely misses come first |
| Sign-off | human | nothing reaches the solver as verified without it; unverified rules are flagged on the card |
| Ingredients of a product | deterministic | RxNorm (NLM); model fallback only behind a flag, only when RxNorm has no data |
| Scheduling, tiers, conflicts | deterministic | Z3 (linear real arithmetic over a circular 24-hour day), encoded in C++ against Z3's native API |
| Checking every schedule shown | deterministic | an independent verifier in exact rational arithmetic |

The model never sees patient data, and the solver never calls a model.

## Evaluation

### Solver (500 random regimens, seed 1, twice: on synthetic days and on real NHANES days)

`python -m linnet.evaluate` and `python -m linnet.evaluate --nhanes-routines`

| Measure | Synthetic days | Real NHANES days |
|---|---|---|
| Greedy heuristic false alarms (gave up; LINNET found a schedule) | **64 of 194 schedulable (33%)** | **67 of 199 schedulable (34%)** |
| Greedy found a schedule but LINNET did not | 0 | 0 |
| Exhaustive 15-minute grid search disagreeing with Z3 | **0** (274 searched) | **0** (265 searched) |
| Conflicts shown that are not minimal | **0 of 78** | **0 of 93** |
| Checks where a solve hit its 10 s limit (shown as undetermined, never as an answer) | 1 | 0 |
| Time per full check (median / p95), C++ solver core | 35 ms / 451 ms | 37 ms / 544 ms |

Both sweeps ran at the same time on one laptop, so the times are pessimistic. Every schedule LINNET returned was also re-checked by the independent verifier. The false-alarm regimens are saved in `data/eval/false_alarms*/`.

Two real NHANES participants, unmodified (#102924 and #98848, both on colestipol and levothyroxine), are false alarms for the greedy heuristic on their recorded day; LINNET schedules both.

Across NHANES 2017–March 2020, 362 adults with a usable routine take a library product. 41 of them (11%) cannot follow their labels on their recorded day as they live (tier 1), with nothing added:
- 36 levothyroxine users ate breakfast less than 30 minutes after their usual wake time; the label asks for it 30–60 minutes before breakfast.
- 4 alendronate users did the same; its label asks for it at least 30 minutes before the first food of the day.
- One takes nimodipine every 4 hours, which needs 20 waking hours.

Wake time is the usual weekday time; breakfast is from the recall day.

Found by this evaluation: Z3 keeps state across solves in one process, and late in a long sweep a few regimens run far slower than from a clean start (with the C++ core, one took 0.04 s fresh and over 40 s in the sweep). The app therefore runs each check in a fresh worker process with a 30 s hard limit, and returns the same answer every time. A solve that still hits its limit is reported as undetermined, or as tier 0 or 1 only if a later solve finds an actual schedule; it is never read as a conflict.

### Rule extraction (pilot: 7 labels)

`python -m library.compare`. Gold = the 11 curated library rules citing the same labels (timing rules and daily maxima). Each gold rule is matched to at most one proposed row; a maximum counts only if it names the right ingredient.

| Run | Gold rules exactly right | Rows rejected by checks | Dosing / limit rows accepted |
|---|---|---|---|
| Claude Haiku 5.5 (API) | 9 / 11 | 4 / 60 | 27 / 15 |
| Haiku stand-in A | 9 / 11 | 13 / 66 | 35 / 9 |
| Haiku stand-in B | 9 / 11 | 13 / 75 | 40 / 9 |
| Sonnet stand-in | 9 / 11 | 14 / 67 | 33 / 9 |

Stand-ins are Claude Code subagents answering the same 21 prompts as the API run, used to measure variation without API spend. Only the first row is a real API call.

All four runs miss the same two rules:
- **Nimodipine "60 mg every 4 hours"**: rejected by design. "Every N hours" is the regimen, and the prescription supplies it; the solver gets the same 4-hour spacing from the prescription. The sentence still reaches the reviewer as a dosing row.
- **Levothyroxine "4 hours before or after drugs known to interfere"**: the sentence names no drugs (they are listed in section 7.1). The runs answered "every other drug", which the checks reject. The sentence stays at the top of the review queue.

Other results:
- On the codeine label, every run tied each maximum to its ingredient (codeine phosphate 360 mg, acetaminophen 4000 mg) and found all three strength-specific regimens (300/15 and 300/30 mg: 1–2 tablets; 300/60 mg: 1 tablet).
- Repeatability: Haiku runs gave a label the same set of timing rules and maxima in 81% of pairwise comparisons.
- The full extraction of all 7 labels (21 calls) on Haiku 5.5 cost about **$0.02**.
- An earlier prompt version (archived in `data/drafts/responses-v1/`) scored on 10 gold rules without ingredient matching, so its numbers are not comparable.

The extraction numbers are a pilot: 7 labels, and the gold rules are partly signed off. Treat them as indicative.

## Prior art

- Tsai & Liu, *Algorithms for Scheduling Interaction Medications* (Academia Sinica TR-IIS-08-001, 2008): heuristics that can report a schedulable regimen as infeasible. Our greedy baseline is a reimplementation in that style, used for comparison.
- Pharmacy automation patents (e.g. US9694966) assign times through a manual interface.

LINNET differs in being complete (if a schedule exists at the stated rules, it finds one), in proving infeasibility at the loosest reading, in tracing every rule to a label sentence, and in giving minimal conflicts and smallest changes.

## Limitations

- Reasons only over rules stated on labels; it is not an interaction database and does not judge clinical appropriateness.
- Deliberately makes no drug recommendations. It shows that a conflict exists and which label sentences cause it; whether to hold, replace or re-time a drug is the prescriber's decision.
- Small library (10 products). Adding a product means extracting and signing off its label.
- Weight-based (mg/kg) doses are extracted and shown for review, but the solver does not use them.
- Tier-1 weights (how bad is moving a meal vs waking earlier) are judgement calls the pharmacist sets; defaults are placeholders.
- Demo patients are constructed. Not for clinical use.

## Privacy

Patient data is solved in memory and never written or logged. The language model reads only public FDA labels. The intake form asks for no name, date of birth or record number. Built with HIPAA in mind; compliance depends on how a pharmacy deploys it.

## Run it

```bash
py -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m uvicorn linnet.api:app --port 8000      # UI at http://localhost:8000
.venv/Scripts/python -m linnet data/patients/tier2-nimodipine-colestipol.json --compare-baseline   # CLI
.venv/Scripts/python -m pytest                                   # tests
```

The solver core is C++ (`solver/native/linnet_solver.cpp`). It is compiled on first use and needs a C++17 compiler: `g++` or `clang++` on PATH (MSYS2 on Windows), or set `CXX`. To build it by hand: `.venv/Scripts/python -m solver.native.build`. Z3's headers and library come from the `z3-solver` wheel.

Extraction (optional, needs `ANTHROPIC_API_KEY` in `.env`):

```bash
.venv/Scripts/python -m library.extract run <ingredient> <spl_set_id> --model claude-haiku-5-5
```

## Layout

| Path | What |
|---|---|
| `solver/` | Z3 encoding in C++ (`native/`), its Python binding, independent verifier |
| `library/` | label schema and loading, extraction, checks, review queue, sign-off store, evaluation |
| `linnet/` | patient input, tiers, conflicts, greedy baseline, API, solver evaluation |
| `web/` | the single-page UI |
| `data/` | labels, curated rules, products, groups, demo patients, extracted drafts, evaluation output |

MIT licensed.
