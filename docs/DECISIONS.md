# Design decisions

Each of these was made against a real failure, most of them found by running the
system rather than reading it.

---

## 1. Never estimate a price locally

**Decision.** If a price cannot be retrieved live, report that it is unavailable.
Never compute, approximate, or fall back to a table.

**Why.** The earlier implementation returned `$59–99` for a retail clinic and
`$100–200` for urgent care whenever the API key was missing *or any request
failed*. The output was indistinguishable from a retrieved price. A user reading
"$59–99" had no way to know the system had simply not asked anyone.

For someone deciding whether they can afford to be seen, that is not a helpful
approximation. It is a number that changes a health decision and has no basis.

**Consequence.** The interface shows *"price not available"* more often than a
system that guesses. That is the correct trade.

---

## 2. Never invent a facility

**Decision.** If a facility search returns nothing, return nothing.

**Why.** The earlier implementation returned synthetic entries — `"CVS MinuteClinic"`,
`"Local Hospital ER"`, `"Local Urgent Care"` — with invented ratings and prices,
whenever geocoding or search failed. In the two-cycle evaluation, nine cases would
have received fabricated clinics.

**Consequence.** Sparse areas legitimately return nothing. Tonopah, Nevada has two
clinics within 20 miles, and the app says two. That is the honest answer, and it
makes the telehealth options the useful part of the response rather than an
afterthought.

---

## 3. A failed safety check is not a clear result

**Decision.** The emergency gate returns three states: `EMERGENCY`,
`NOT_EMERGENCY`, `UNKNOWN`. `UNKNOWN` never proceeds to the rest of the pipeline.

**Why.** The gate previously caught every exception and returned `False` — "not an
emergency". An endpoint timeout therefore routed a patient with chest pain to a
retail clinic. This contradicted the gate's own prompt, which instructs the model
to answer YES when uncertain.

Flipping it to fail closed is not defensible either: a JSON parse error would tell
someone with a headache to call 911, and a system that cries wolf gets ignored.

**Consequence.** `UNKNOWN` produces an amber panel carrying the actual 911 criteria
— chest pain, trouble breathing, one-sided weakness, confusion, heavy bleeding —
so the patient can make the call themselves with real information.

---

## 4. The emergency gate lives outside the orchestrator

**Decision.** MedGemma's emergency check runs before the graph is even constructed.

**Why.** It answers the only question in the system where being wrong is dangerous
rather than expensive. Putting it inside an orchestration graph means a graph
failure could swallow it.

**Consequence.** A completely dead orchestrator still produces "call 911" for a
cardiac presentation. There is a test for exactly this.

---

## 5. Judgement from the model, guarantees from the code

**Decision.** The orchestrator decides which facilities are appropriate. It can
only choose among facilities the tool actually returned.

**Why.** Filtering OpenStreetMap results needs judgement — no tag distinguishes a
walk-in clinic from a chiropractor, and `healthcare=urgent_care` is essentially
unused in practice. But a model given free rein over the output could invent a
clinic that sounds plausible.

**Consequence.** `filter_facilities` matches returned `place_id`s against the ones
supplied and silently drops anything else. The model can reject; it cannot add.

---

## 6. Structured output from the API, not from a parser

**Decision.** Pydantic models are passed to Gemini as `response_schema`.
LangChain's output parsers were evaluated and not used.

**Why.** A prompt-based parser asks the model for JSON and hopes. `response_schema`
constrains generation at the API, so malformed output is not possible.

This mattered concretely. A `Dict[str, str]` field for rejection reasons came back
empty on every live call, because Gemini's schema subset cannot express an object
with arbitrary keys — the field was dropped silently rather than erroring.
Restructuring it as a list of objects with required fields fixed it. A
prompt-based parser would have produced the same silent gap with no signal.

**Consequence.** LangGraph is used for the graph and the retry cycle. LangChain is
not a dependency.

---

## 7. Cap the results after judgement, not before

**Decision.** The orchestrator sees every candidate found. Only the surviving
nearest three are priced and shown.

**Why.** An earlier version capped to three before the filter ran. The orchestrator
therefore received a pre-truncated list: a specialty practice inside the nearest
three was never rejected, and a real clinic just outside them was never seen. The
filter existed but could not act — the symptom was an empty rejection log on a run
that should have rejected four facilities.

**Consequence.** More candidates are judged than are displayed, and the number
dropped is recorded rather than disappearing.

---

## 8. Telehealth is not gated by the clinical model alone

**Decision.** MedGemma's `telehealth_coverage` informs the choice. It does not
decide it.

**Why.** As a hard gate, it suppressed every online option for pink eye — the
canonical remote-treatable complaint, which Amazon Clinic advertises treating. Even
demoted to a hint, the orchestrator deferred to it almost completely.

The clinical model is systematically cautious about remote care and never sees the
provider list. It is one opinion, not a rule.

**Consequence.** Pink eye now surfaces Amazon Clinic at $35–75 alongside urgent care
at $150–250. The system still refuses telehealth for a laceration needing sutures,
so the discrimination is real rather than a blanket reversal.

---

## 9. The review node is the evaluation instrument

**Decision.** The orchestrator's review step records what it rejected and what it
found missing, per case, in the output.

**Why.** Evaluating an agentic pipeline usually means a separate analysis pass.
Since the review node already reasons about whether the answer is complete, its
findings are the analysis — for free, on every request, including in production.

**Consequence.** Every result carries its own account of what was excluded and why.
The demo screenshots show it in a collapsed panel; the eval harness aggregates it.

---

## 10. Overpass, not Nominatim, for finding places

**Decision.** Nominatim geocodes the ZIP. Overpass finds what is near it.

**Why.** Nominatim is a geocoder. Querying it for `"urgent care, 95134, USA"` is a
free-text *name* search in which the ZIP is not a constraint — it returned clinics
in the Bronx and Los Angeles for a San Jose query. The distance filter then
correctly discarded all of them, so four of five test ZIPs returned nothing at all.

The data existed the whole time. We were asking the wrong way.

**Consequence.** Facilities found per ZIP went from `1 / 0 / 0 / 0 / 0` to
`10 / 10 / 10 / 10 / 2`. The trailing 2 is Tonopah, which genuinely has two.
