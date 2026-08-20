# Fixing the Retail Clinic collapse

**Scope:** finding #2 of `docs/EVALUATION.md` — all five Retail Clinic cases escalated to
Urgent Care, in both cycles. This is a proposal, not a patch. Nothing here changes the
emergency gate.

Findings #1 (facility search) and #3 (price band) look already addressed in the working
tree: `services/maps_service.py` now splits Nominatim-for-geocoding from Overpass-for-
proximity, and `PriceEstimate` in `agents/contracts.py` carries `low_usd`/`high_usd` rather
than the floor. Finding #4 (`likely_conditions` empty) is untouched and out of scope here.

---

## What the numbers actually say

Two cycles, 20 cases, temperature 0.1:

| | |
|---|---|
| Care-level accuracy | 14/20 both cycles |
| Cases differing between runs | 0/20 |
| Under-triage across 40 case-runs | **0** |
| Retail Clinic | **0/5**, all escalated to Urgent Care |

Zero variance is the load-bearing fact. At 0/20 differing, this is not a model that is
uncertain and landing on the cautious side of a coin flip. It is a model reasoning
*consistently* to the wrong answer. That makes 70% a property of the prompt, and it makes
the six failures reproducible targets — which in turn means a prompt change can be
evaluated on 20 cases and believed, without averaging over seeds.

It also rules out the sampling-based fixes. Do not lower temperature further, and do not
raise it to "explore" — that would trade a reproducible failure for an irreproducible one
and destroy the eval's main virtue.

---

## Diagnosis

Four things in `triage.py` push the Retail Clinic / Urgent Care boundary upward. They
compound, and none of them is a model capability problem.

### 1. The three care settings are never defined to the model

`ASSESS_PROMPT` introduces them only as an output-format constraint:

```
- care_setting: string — exactly one of "Home Care", "Retail Clinic", "Urgent Care"
```

That is the entire specification. The model gets three bare labels and must infer what
separates them from the English words alone. Between two adjacent labels with no stated
discriminator, an unmodified clinical prior lands on the more capable setting — which is
exactly the observed behaviour.

Discriminating criteria *do* exist in this repo, in `agents/schemas.py`:

```python
RETAIL_CLINIC = 2   # Prescription, known conditions (strep, UTI, pink eye)
URGENT_CARE  = 3   # Diagnostic tests, unknown conditions
```

Those are Python comments in a module `triage.py` does not import. They never reach the
model. The five failed cases are pink eye, UTI, strep, ear infection and rash — four of
which are named or implied in a comment sitting two files away from the prompt.

### 2. Phase 2 wears the emergency gate's persona

Both prompts open with the identical line:

```
You are a clinical decision support system trained on emergency medicine triage protocols.
```

For `EMERGENCY_GATE_PROMPT` that is correct. For `ASSESS_PROMPT` it is backwards: this call
runs *after* the patient has been cleared, and the prompt says so. Emergency medicine triage
protocols are constructed to escalate under uncertainty — that is what they are for. Asking
a model wearing that persona to pick the cheapest adequate outpatient setting is asking it
to work against its stated role.

`docs/EVALUATION.md` guessed the inherited element was the gate's `When uncertain, answer
YES` rule. Worth correcting: that string is *not* in `ASSESS_PROMPT`. What carries over is
the persona line, which is a narrower and more directly fixable target.

### 3. The reasoning order primes escalation, and isn't honoured anyway

```
Reason through what labs, diagnostics, and physical interventions this patient would need.
From that, determine the appropriate care setting.
```

Enumerate-diagnostics-first is a reasonable chain, but under an emergency-medicine persona
the path of least resistance is to list *something*, and the setting is then derived from
whatever got listed.

Except the derivation isn't actually happening. `docs/EVALUATION.md` records case 6 (pink
eye) returning **empty** `labs` and **empty** `interventions` and still answering Urgent
Care. The model named no requirement that a retail clinic could not meet, then escalated
anyway. The stated rule is decorative — which is worth knowing, because it means adding
criteria to that rule only helps if the rule is also made checkable.

### 4. The one place that decides cost is the one place that isn't told about cost

Cost-consciousness is encoded throughout the repo. `services/maps_service.py` refuses to
satisfy a Retail Clinic request with an Urgent Care facility, and says why in a comment:
*that is the more expensive tier*. `PRICING_PROMPT` asks for realistic self-pay bands.

`ASSESS_PROMPT` — which picks the tier and therefore determines the bill — contains no
mention of cost, no ordering over the three settings, and no instruction to prefer the
least costly adequate one. Separately, `care_settings` (plural) invites answers like
`["Retail Clinic", "Urgent Care"]` with no stated tie-break for which goes in the singular
`care_setting` field that `run_eval.py` actually scores.

---

## The constraint that comes first

**Zero under-triage across 40 case-runs. Do not trade it away.**

Over-triage is the safe failure direction. It costs an uninsured patient roughly $100 — the
gap between a $60–100 retail visit and a $150–250 urgent care one. Under-triage costs them
a missed fracture, a missed pyelonephritis, a missed sepsis. These are not commensurable,
and a change that fixes Retail Clinic while introducing a single under-triage is worse than
shipping nothing.

Concretely, accept a variant only if, across both cycles:

- ER stays 5/5
- Urgent Care stays 5/5
- no Home Care case moves *up* into a paid visit
- **count of under-triaged cases stays exactly 0**

Retail Clinic at 3/5 with those held is a win. Retail Clinic at 5/5 with one Urgent Care
case lost is a rollback.

---

## Strategy 1 — Define the boundary by capability, and state the objective

**Do this one first, on its own.**

Three edits to `ASSESS_PROMPT`, all in `triage.py`:

**(a) Drop the emergency-medicine persona from Phase 2 only.** Leave
`EMERGENCY_GATE_PROMPT` exactly as it is. Something like *"You are a clinical decision
support system that routes cleared, non-emergency patients to the appropriate outpatient
setting."*

**(b) Define the settings by what the facility can physically do.** Not by example
conditions — by capability, so it generalises past the five cases in `eval_data.json`:

- **Retail Clinic** — nurse-practitioner staffed walk-in. History and physical exam;
  point-of-care testing (rapid strep, rapid flu/COVID, urinalysis, urine culture, blood
  glucose); prescribing, including antibiotics. Treats uncomplicated presentations with a
  recognisable pattern.
- **Urgent Care** — everything a retail clinic does, plus on-site imaging (X-ray), IV
  fluids, laceration repair and suturing, nebulised treatment, splinting and fracture care,
  venous blood draws beyond point-of-care panels, foreign-body removal.
- **Home Care** — self-limiting, manageable with OTC treatment and time; no prescription and
  no examination changes management.

**(c) State the decision rule and the tie-break.**

> Choose the *least costly* setting that can deliver every lab and intervention you listed.
> If every item on your list can be performed at a retail clinic, answer Retail Clinic.
> Escalate to Urgent Care only when you have named a specific lab or intervention that a
> retail clinic cannot perform. `care_setting` must be the least costly member of
> `care_settings`.

**Why this targets the observed failure.** The failures are deterministic, so the model is
not guessing — it is reasoning correctly from a specification that doesn't contain the
distinction it's being scored on. (b) supplies the missing discriminator, (a) removes the
persona pushing it upward, (c) supplies the objective and closes the plural-field ambiguity.
Every one of the five failed cases is resolvable from point-of-care testing plus a
prescription, which is precisely the capability line (b) draws.

**Why it's the cheap first move.** Prompt-only. No deployment change, no cost change, no
new failure modes, and one eval run — 20 cases, 2 calls each — tells you whether it worked.

---

## Strategy 2 — Boundary-specific few-shot anchoring

**Only if Strategy 1 leaves Retail Clinic below 4/5, or softens Home Care.**

Add 5–6 worked examples to `ASSESS_PROMPT`, each showing presentation → labs and
interventions → setting, so the model sees the criterion applied rather than only stated.

Two constraints matter more than the examples themselves.

**Hold the examples out of the eval set.** `eval_data.json` cases 6–10 are pink eye, UTI,
strep, ear infection and rash. If the few-shots use those presentations, a subsequent 5/5
measures memorisation and tells you nothing about the boundary. Use adjacent-but-distinct
presentations: ten days of facial pain and congestion; a tick that needs removal; an
uncomplicated cold sore; a routine prescription refill; athlete's foot.

**Make the anchoring bidirectional.** At least two examples must land on Urgent Care — an
ankle injury that cannot bear weight (needs X-ray), a gaping laceration through the dermis
(needs suturing) — and at least one on Home Care. One-sided examples at a boundary teach a
*direction*, and a model taught to de-escalate at this boundary will carry that into
Urgent Care cases. That is the mechanism by which this strategy could introduce the
under-triage the constraint above forbids. Two-sided examples teach the *criterion*, and
the Urgent Care examples are what hold the safety floor.

Cost: roughly 400–600 additional input tokens on every Phase 2 call. `MAX_SYMPTOM_CHARS`
already caps the variable part at 600, so the prompt stays bounded.

---

## Not proposing: two-stage classification

A coarse pass plus a focused second call at ambiguous boundaries would probably work, but it
doubles Phase 2 cost, adds a routing decision that can itself be wrong, and introduces a
second prompt to keep calibrated. The evidence points at an underspecified prompt rather
than an inability to discriminate — the model has the clinical knowledge, it was never told
what it was choosing between. Spend that complexity only if Strategies 1 and 2 both fail.

## Guardrail: make the stated rule checkable (ships with either strategy)

The prompt tells the model to derive the setting from the labs and interventions it named.
Nothing verifies that it did. Case 6 escalated on an empty list, twice.

```python
unjustified = (
    not assessment.labs
    and not assessment.interventions
    and assessment.care_setting == "Urgent Care"
)
```

**Flag it; do not auto-downgrade on it.** Silently moving a patient down a tier on a rule
with no clinical input is exactly the kind of change that manufactures under-triage. Its
value is as an instrument: it costs nothing, needs no model call, and answers whether the
prompt's own reasoning rule is being followed. Surface it in `review_notes` and count it in
the eval.

---

## Measurement

**Langfuse is not currently wired in** — no `langfuse` in `requirements.txt`, no
`LANGFUSE_*` in `.env.example`, no client anywhere in the tree. So this is an addition, and
it is small: `VertexLLMClient.complete()` in `services/llm_client.py` is the only path to
the model. Wrapping that one method turns every Phase 1 and Phase 2 call into a span.

**Per generation:** `phase` (`gate` | `assess`), `prompt_variant`
(`baseline` | `v2-criteria` | `v2-criteria-fewshot`), `cycle`, rendered prompt, raw
completion, latency, token counts.

**Per case, as a trace:** `case_id`, `category`, `expected`, `actual`, `passed`, `labs`,
`interventions`. `EvalResult` in `run_eval.py` already carries all seven — this is a mapping
job, not new collection.

**Scores to attach:**

| Score | Definition | Target |
|---|---|---|
| `care_level_accuracy` | `passed`, 0/1 | ≥ 17/20 |
| `triage_direction` | `LEVEL[actual] - LEVEL[expected]`, signed | **count of negatives = 0** |
| `retail_boundary_hit` | accuracy restricted to `expected == "Retail Clinic"` | ≥ 3/5, target 5/5 |
| `unjustified_escalation` | the guardrail flag above | trending to 0 |

`triage_direction` is the one the repo is missing. `run_eval.py` records only pass/fail, so
over- and under-triage are indistinguishable in the output — the safety property that makes
the current 70% acceptable is invisible to the harness that produced it. Signed direction
makes it a number you can chart and gate on.

**Protocol.** Run 2 cycles per variant, as before, tagging `cycle` so the zero-variance
property stays visible. If variance *appears* after a change, that is a finding in itself:
it would mean the new prompt made the decision marginal rather than confident, and a variant
that is right on average but unstable is not obviously better than one that is wrong and
stable.

**Ship gate.** Across both cycles: negatives on `triage_direction` = 0, ER 5/5, Urgent Care
5/5, Retail Clinic ≥ 3/5.

**One caveat on the numbers.** Twenty cases, five per tier, means the whole Retail Clinic
result rests on five items — a move from 0/5 to 4/5 is four cases. Given zero variance
that is real signal rather than noise, but it is not a production accuracy estimate.
Widening the Retail Clinic and Urgent Care tiers to ~10 cases each would make the boundary
metric mean considerably more, and it is cheap: two model calls per case, no deployment
change, and `run_eval.py --category "Retail Clinic" "Urgent Care"` already selects exactly
that slice.
