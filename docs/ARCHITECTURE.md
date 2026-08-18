# Architecture

## Components

| Component | Role | Runs on |
|---|---|---|
| **MedGemma 27B** | Emergency gate, clinical assessment | Vertex AI, `g4-standard-48` + 1× RTX PRO 6000 |
| **Gemini Flash** | Orchestrator, facility filtering, pricing, review | Vertex AI, `gemini-3.6-flash` |
| **Overpass** | Facility discovery by proximity | OpenStreetMap, free, no key |
| **Nominatim** | ZIP → coordinates only | OpenStreetMap, free, 1 req/sec |
| **LangGraph** | Graph execution and the retry cycle | local |
| **Pydantic** | Contracts, and the API-level output schema | local |

No vector store, no RAG, no cached price data. Every facility and every price is
fetched at request time.

## Flow

### Stage 1 — Emergency gate (outside the graph)

MedGemma answers one binary question. Three outcomes:

- `EMERGENCY` → return immediately with 911 guidance. No facility search, no pricing.
- `UNKNOWN` → the check did not complete. Return an amber panel with the 911 criteria.
- `NOT_EMERGENCY` → continue.

This runs before the graph is constructed, so no orchestration failure can reach it.

### Stage 2 — Clinical assessment

MedGemma returns a validated `Assessment`:

```python
care_setting        # "Home Care" | "Retail Clinic" | "Urgent Care"
labs                # ["Rapid influenza test", "CBC"]
interventions       # ["IV fluids", "Splinting"]
likely_conditions   # differential, not shown to the patient
telehealth_coverage # "yes" | "no" | "maybe"  — advisory
reasoning           # one or two sentences, patient-facing
recommendation      # one sentence
```

An unrecognised `care_setting` raises rather than defaulting. Every historic
default here resolved to Retail Clinic, which quietly downgraded "ER" and every
typo alike.

`"ER"` is not a valid Phase 2 answer — the gate owns that question, and Phase 2
only runs on patients it has cleared.

### Stage 3 — Orchestration graph

```
find ──> filter ──> price ──> telehealth ──> review
           ^                                    │
           └────────── bounded retry ───────────┘
```

**find** — Nominatim geocodes the ZIP; Overpass returns medical features within
20 miles. Level 1 (Home Care) skips this entirely.

**filter** — Flash judges which candidates can treat this presentation, and
rejects the rest with a reason each. No search grounding: this is judgement, not
a web question. Only `place_id`s that were supplied can survive.

**price** — one grounded Flash call per surviving facility, run concurrently.
Names the specific labs and interventions from the assessment, so an x-ray and a
splint are not priced as a walk-in consultation. Returns a band, never a single
figure taken from the bottom of a range.

**telehealth** — Flash selects providers that genuinely fit and says why. Runs
even when no facilities were found, because that is exactly when it matters.

**review** — Flash judges whether the answer delivers all three promises. Runs at
`thinking_budget=0` — it is a verdict, not a deliberation. If something is
recoverable it names the tool to retry; the loop is capped at two.

### Stage 4 — Result

A flat dict consumed by both the CLI and the web UI. Facilities carry
`latitude`/`longitude` so they can be mapped, and `cost_basis` so a figure can be
traced on screen.

## Failure behaviour

| Failure | Behaviour |
|---|---|
| Emergency gate unreachable | `UNKNOWN` — 911 criteria shown, pipeline stops |
| Assessment unparseable | Error surfaced; never a defaulted care level |
| Orchestrator dead | Clinical answer still returned; `review_notes` says places and prices are missing |
| Overpass rate-limited | Retried with backoff; then an honest empty list |
| A price lookup fails | That option is shown unpriced; the others still price |
| No facilities nearby | Reported as such; telehealth becomes the answer |

The pattern throughout: degrade to a smaller true answer, never to a plausible
false one.

## Testing

141 tests, all offline — no credentials, no endpoint, no network. Dependencies
are injected, so the graph runs end to end against stubs.

The MedGemma endpoint is deployed only for evaluation runs and destroyed
immediately after. A 27B model on a GPU node costs roughly $5/hour, and about 29
minutes of any run is deployment before the model serves a single token.
