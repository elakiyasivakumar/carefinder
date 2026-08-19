# Design trade-offs

Every row here was a real fork. The "instead of" column is what was actually
considered, and in several cases actually built and then replaced.

## Stack choices

| Chose | Instead of | Why | What it cost |
|---|---|---|---|
| **LangGraph** | LangChain | The pipeline needs a **cycle**: the review node can send work back to pricing. Chains are acyclic; expressing "retry that tool, then re-review" in a chain means hand-rolling the loop and the state. LangGraph models it directly with a conditional edge. | A dependency, and a graph abstraction over what could be a `while` loop. Justified only because of the cycle. |
| **Pydantic + `response_schema`** | LangChain `StructuredOutputParser` | The parser is **prompt-based** — it asks for JSON and parses what returns. Gemini's `response_schema` constrains generation at the API, so malformed output is not possible. | Locks the orchestrator to providers with native schema support. A parser would have been portable. |
| **MedGemma 27B** | MedGemma 4B | 4B fits on one L4 and is far cheaper. 27B was chosen for clinical reasoning quality on differential-style questions. | ~54GB VRAM instead of ~8GB, which forces a data-centre GPU and removes on-device deployment. See [edge](#could-this-run-on-the-edge). |
| **Overpass** | Google Places | Places has hours, phone numbers, ratings and better coverage. Overpass is free, needs no key, and returns the same OSM data the map tiles come from. | Thinner metadata, no opening hours, and a public instance that rate-limits under load. |
| **Gemini Flash orchestrator** | Deterministic Python routing | OSM has no usable "urgent care" tag, so no rule can separate a walk-in clinic from a chiropractor. A model can. | Non-determinism in a path that was previously auditable, and ~$0.11/query. |
| **Vertex AI Gemini** | Gemini API key | The free-tier key returned `RESOURCE_EXHAUSTED` mid-evaluation. Vertex has no free-tier cap. | Requires a billed GCP project; no zero-cost path for a contributor. |
| **Flash + search grounding** | Perplexity Sonar | One provider instead of two, one fewer key, and grounding composes with `response_schema` so pricing returns a typed band directly. | Ties pricing to Google's index. Grounding is **97% of per-query cost**. |

## Iterations — what was built, then replaced

| # | Built | Broke because | Replaced with |
|---|---|---|---|
| 1 | Local price table as a fallback | Returned `$59–99` whenever a key was missing *or any request failed*. Indistinguishable from a retrieved price. | Retrieve or report unavailable. Never estimate. |
| 2 | Synthetic facilities when search failed | Invented "Local Hospital ER" with a fake rating. 9 of 11 eval cases would have received fabricated clinics. | Return `[]`. An empty list is a true answer. |
| 3 | Gate returning `bool` | `except: return False` meant a timeout routed chest pain to a retail clinic. | Tri-state verdict. `UNKNOWN` never proceeds. |
| 4 | `CARE_SETTING_TO_LEVEL.get(x, 2)` | Silently turned `"ER"` — and every typo — into Retail Clinic. | Strict lookup that raises. |
| 5 | Nominatim free-text facility search | A name search ignores the ZIP. San Jose returned clinics in the Bronx and LA; 4 of 5 ZIPs returned nothing. | Overpass proximity query. Facilities per ZIP went `1/0/0/0/0` → `10/10/10/10/2`. |
| 6 | Keyword filter over OSM results | Surfaced a chiropractor, a medspa and a paediatric office as urgent care. | Flash filter with per-rejection reasons. |
| 7 | `Dict[str, str]` for rejection reasons | Gemini's schema subset cannot express arbitrary-key objects. The field came back **empty on every call**, silently. | `List[RejectedFacility]` with required fields. |
| 8 | Cap results to 3 before filtering | The orchestrator received a pre-truncated list, so it could not reject anything outside the top 3. Symptom: an empty rejection log on a run that should have rejected four. | Cap after judgement. |
| 9 | `telehealth_coverage` as a hard gate | Suppressed every online option for pink eye — the canonical remote case. | Advisory hint. Pink eye now surfaces a $35 option beside a $250 one. |
| 10 | Sequential price lookups | Three grounded searches in series, ~57s per query. | Concurrent. 57s → 32s. |
| 11 | Sampling params in `parameters={}` | Model Garden's vLLM container reads them from the instance. `temperature=0.1` was **silently ignored**. | Params inside the instance. |
| 12 | Raw completion prompts to MedGemma | An instruction-tuned model given a raw prompt emits EOS immediately. Every call returned an empty string. | Gemma turn markers. |

Items 7, 11 and 12 share a shape worth naming: **all three failed silently**, and
none was findable offline. Each was caught only by running the real thing and
noticing an output that was plausible but empty.

## Cost per query

Measured, not estimated. One triage = 2 MedGemma calls + ~6 Flash calls.

| Component | Tokens | Cost |
|---|---|---|
| MedGemma gate + assessment | ~613 in, ~465 out | GPU node time — see below |
| Flash filter / telehealth / review | ~995 in, ~400 out | ~$0.001 |
| Flash pricing ×3 | ~363 in, ~540 out | ~$0.002 |
| **Search grounding ×3** | — | **~$0.105** |
| Overpass + Nominatim | — | free |
| **Flash total** | | **~$0.108** |

**Grounding is 97% of the marginal cost.** Token spend is rounding error. The
lever that matters is how many facilities get priced — `MAX_PRICED = 3` is a cost
decision as much as a UX one, and dropping to 2 cuts the bill by a third.

### MedGemma is not a per-query cost

It is a **step function**. `g4-standard-48` + 1× RTX PRO 6000 bills roughly
$5–6/hour whether it serves one request or a thousand, and about **29 minutes of
every cold start is deployment** before the first token.

| Utilisation | Effective cost per query |
|---|---|
| 1 query on a cold endpoint | **~$3–4** (36 min of node time) |
| Steady load, vLLM batching at 16 concurrent | **~$0.005** |
| Always-on, no traffic | ~$3,600/month |

This is the real economics of the project: the clinical model is either nearly
free or ruinously expensive, entirely depending on utilisation, and there is no
middle. Every evaluation in this repo deploys, runs, and destroys the endpoint
inside one window for exactly that reason.

## Processing requirements

| | MedGemma 27B | MedGemma 4B |
|---|---|---|
| Weights (bf16) | ~54 GB | ~8 GB |
| 4-bit quantised | ~14 GB | ~2.5 GB |
| Minimum accelerator | 1× 80–96 GB (A100 80GB, H100, RTX PRO 6000) or 4× L4 | 1× L4 (24 GB), or a laptop GPU |
| Cold start | ~29 min | ~10 min |

Default Vertex serving quota is commonly **0** for A100 and H100 and **2** for
L4, which leaves `g4-standard-48` with one RTX PRO 6000 as the only configuration
that deploys without a quota request.

## Could this run on the edge?

**Partly — and the part that could is the part that should.**

The clinical model at 4B, quantised to 4-bit, is ~2.5 GB. That runs on a current
phone NPU or any laptop. The 27B does not and will not.

But the retrieval cannot be local. Facility search needs OSM, pricing needs a live
web search, and neither can be shipped in an app bundle without becoming the
stale cached data this system exists to avoid.

So the honest architecture is **hybrid, split along the privacy boundary**:

| Runs on device | Runs in the cloud |
|---|---|
| Emergency gate | Facility search (ZIP only) |
| Clinical assessment | Pricing (facility name + procedure names) |
| **Symptoms never leave the phone** | Nothing identifying is sent |

That split is more than an optimisation. Free-text symptoms are the sensitive
part of the input; ZIP and "rapid influenza test" are not. Keeping inference local
means the most personal field never transits a network, while the queries that do
go out carry nothing that identifies a patient.

The cost of that design is the accuracy gap between 4B and 27B, which this
project has not measured. Given 27B already over-triages every retail-clinic case,
a smaller model plausibly does worse on exactly the distinction that matters most.
That is the experiment worth running before treating on-device as viable.
