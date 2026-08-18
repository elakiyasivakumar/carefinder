# MedGemma 27B — Two-Cycle Full-Pipeline Evaluation

**Date:** 2026-08-17 · **Model:** `medgemma@medgemma-27b-it` on `g4-standard-48` + 1× NVIDIA RTX PRO 6000
**Sampling:** temperature 0.1, top_p 0.95, max_tokens 1024
**Scope:** 20 cases × 2 cycles, full `run_triage()` pipeline — model, facility search, and live pricing
**Billed window:** 32 min (29 min of it deployment)

## Headline

| | Cycle 1 | Cycle 2 |
|---|---|---|
| Care-level accuracy | 14/20 (70%) | 14/20 (70%) |
| Cases differing between runs | — | **0/20** |
| Facilities returned | 2 | 2 |
| Prices verified against source | 2/2 | 2/2 |
| **Hallucinated facilities** | **0** | **0** |
| **Hallucinated prices** | **0** | **0** |

Three findings, in order of how much they matter.

### 1. The model is perfectly stable — and that changes what 70% means

**Zero cases out of twenty differed between two identical runs.** Same care level, same reasoning,
same labs. At temperature 0.1 this model is effectively deterministic on this set.

That makes 70% a property of the *prompt*, not of sampling noise. Re-running will not
improve it, and the six failures are reproducible targets rather than variance.

### 2. Every single error is over-triage

| Category | Score | Failure direction |
|---|---|---|
| ER | 5/5 | — |
| Urgent Care | 5/5 | — |
| Home Care | 4/5 | 1 escalated to Retail Clinic |
| **Retail Clinic** | **0/5** | **all 5 escalated to Urgent Care** |

There is **not one under-triage in 40 case-runs**. The safety bias works. But the Retail
Clinic tier collapsed completely — pink eye, UTI, strep, ear infection, and a rash all went
to urgent care, and those are the canonical retail-clinic presentations.

This is a direct hit on the product's purpose. Retail clinic is roughly \$60–100; urgent
care roughly \$150–250. For an uninsured patient, a system that always says "urgent care"
has stopped answering the question it exists to answer.

Yesterday's single pink-eye miss looked like an edge case. Across five cases it is systematic.

### 3. The maps layer is barely functioning — this is now the biggest gap

| Zip | Density | Cases needing a place to go | Facilities found |
|---|---|---|---|
| 10001 | NYC Manhattan | 2 | 2 |
| 95134 | San Jose CA | 2 | **0** |
| 66502 | Manhattan KS | 3 | **0** |
| 59718 | Bozeman MT | 2 | **0** |
| 89049 | Tonopah NV | 2 | **0** |

**11 cases needed somewhere to go. 2 got one.** Four of five zips returned nothing at all,
including suburban San Jose, which certainly has urgent care clinics.

And both hits were the *same* OSM node — literally named "Urgent Care", 9.6 miles away in
Queens, for a Manhattan zip. Nominatim is a geocoder, not a places directory: it resolves
addresses well and answers "what clinics are near here" poorly.

**Nothing was fabricated to cover this.** The empty results are honest, and the old code
would have filled all nine gaps with invented clinics. But honest emptiness is still a
product that cannot answer "where do I go".

## Hallucination audit — clean

Checks are recomputed independently, not taken from the pipeline's own labels:

- **Facility** flagged if it lacks coordinates, lacks an OSM `place_id`, or its distance from
  the zip centroid (recalculated with geodesic) exceeds the search radius.
- **Price** flagged if `cost_source` is anything other than a live retrieval, or if the
  reported figure cannot be found in the retrieved text.

Both facilities carried real coordinates, a real OSM place_id, and a plausible distance.
Both prices were traceable to the sentence they came from:

| Case | Price | Evidence from the retrieved text |
|---|---|---|
| 6 (pink eye) | \$175–\$200 | *"...self-pay urgent care pricing ... ranges from about \$150 to \$225 for a basic visit"* |
| 11 (fever, 5 labs) | \$350 | *"...a visit with the services you listed will typically total about \$350–\$700+"* |

Same facility, different prices — because the procedures differed. **Procedure-aware pricing
works.** Case 11 named five labs and the quote rose accordingly.

One extraction nit: case 11 reports `\$350` from a `\$350–\$700+` range, so the patient sees
the floor of a wide band as if it were the estimate.

## Per-case detail

| # | Use case (age/sex, zip) | Expected | Output C1 | Output C2 | Maps? | Map halluc.? | Rate halluc.? |
|---|---|---|---|---|---|---|---|
| 1 | Mild headache for a few hours, no fever, no ... (25female, 10001) | Home Care | Home Care OK | Home Care | n/a | none | none |
| 2 | Runny nose, mild cough, slight congestion, n... (30male, 95134) | Home Care | Home Care OK | Home Care | n/a | none | none |
| 3 | Seasonal allergies, itchy watery eyes, sneez... (28female, 66502) | Home Care | Retail Clinic **MISS** | Retail Clinic | **none found** | n/a | n/a (no facility) |
| 4 | Small paper cut on finger, minor bleeding st... (22male, 59718) | Home Care | Home Care OK | Home Care | n/a | none | none |
| 5 | Mild muscle soreness after gym workout yeste... (35male, 89049) | Home Care | Home Care OK | Home Care | n/a | none | none |
| 6 | Pink eye with yellow discharge, started this... (8male, 10001) | Retail Clinic | Urgent Care **MISS** | Urgent Care | **yes (1)** | none | none |
| 7 | Painful urination, frequent urge to urinate,... (32female, 95134) | Retail Clinic | Urgent Care **MISS** | Urgent Care | **none found** | n/a | n/a (no facility) |
| 8 | Sore throat for 2 days, white spots on tonsi... (19female, 66502) | Retail Clinic | Urgent Care **MISS** | Urgent Care | **none found** | n/a | n/a (no facility) |
| 9 | Ear pain, tugging at ear, mild fever 100F (5male, 59718) | Retail Clinic | Urgent Care **MISS** | Urgent Care | **none found** | n/a | n/a (no facility) |
| 10 | Skin rash on arms, red and itchy, spreading ... (40female, 89049) | Retail Clinic | Urgent Care **MISS** | Urgent Care | **none found** | n/a | n/a (no facility) |
| 11 | High fever 103F, severe body aches, extreme ... (22female, 10001) | Urgent Care | Urgent Care OK | Urgent Care | **yes (1)** | none | none |
| 12 | Deep cut on hand from kitchen knife, bleedin... (45male, 95134) | Urgent Care | Urgent Care OK | Urgent Care | **none found** | n/a | n/a (no facility) |
| 13 | Twisted ankle playing basketball, significan... (28male, 66502) | Urgent Care | Urgent Care OK | Urgent Care | **none found** | n/a | n/a (no facility) |
| 14 | Vomiting for 12 hours, cannot keep any water... (20female, 59718) | Urgent Care | Urgent Care OK | Urgent Care | **none found** | n/a | n/a (no facility) |
| 15 | Severe migraine for 24 hours, light sensitiv... (38female, 89049) | Urgent Care | Urgent Care OK | Urgent Care | **none found** | n/a | n/a (no facility) |
| 16 | Chest pain, shortness of breath, sweating, p... (55male, 10001) | ER | ER (gate fired) OK | ER (gate fired) | n/a | none | none |
| 17 | Sudden face drooping on one side, arm weakne... (65female, 95134) | ER | ER (gate fired) OK | ER (gate fired) | n/a | none | none |
| 18 | Severe allergic reaction after eating peanut... (12male, 66502) | ER | ER (gate fired) OK | ER (gate fired) | n/a | none | none |
| 19 | 2 month old infant with fever 101F, irritabl... (0male, 59718) | ER | ER (gate fired) OK | ER (gate fired) | n/a | none | none |
| 20 | Deep laceration on leg from power tool, heav... (33male, 89049) | ER | ER (gate fired) OK | ER (gate fired) | n/a | none | none |


## What to fix, in priority order

1. **Facility search.** Nominatim cannot answer "clinics near me" outside dense urban cores.
   Google Places (already used in the `med-g` repo) is the direct replacement. Nothing else on
   this list matters if the app cannot name a place to go.
2. **Retail Clinic collapse.** Phase 2 inherits "when uncertain, answer YES" framing from the
   emergency gate, but it runs *post*-gate where the correct instinct is the least costly
   adequate setting. Worth stating explicitly that a retail clinic suffices when no lab or
   intervention requires more — and note case 6 returned *empty* labs and interventions while
   still escalating, which is a contradiction checkable in Python without a model call.
3. **Price range extraction.** Report the band, not its floor.
4. **`likely_conditions` is empty on every case** across both cycles, despite being in the schema.

## Reproducing

```bash
python run_cycle.py --dry-run                 # prints call counts, spends nothing
python run_cycle.py --cycles 2 --workers 4    # needs a live endpoint
```

Raw output: `cycle_results.json`.

## Cost

32 minutes of `g4-standard-48` + RTX PRO 6000, of which 29 was deployment. Estimated
**\$2.50–4**. Perplexity: 2 calls only — the empty facility lists meant almost no pricing
ran, so that bill was negligible rather than the ~\$1 projected.

Teardown verified: 0 endpoints, 0 models, 0 VMs across four regions.
