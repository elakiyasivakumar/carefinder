# CareFinder

**An agentic triage system that tells uninsured patients where to go and what it will cost — before they leave home.**

Built on MedGemma 27B for clinical reasoning, a Gemini Flash orchestrator for
tool coordination, and OpenStreetMap for real facility data. Every number shown
to a patient is retrieved live. Nothing is estimated locally.

---

## The problem

If you are uninsured in the United States and wake up with a fever, you face two
questions and can answer neither:

1. **Where should I go?** Home, a retail clinic, urgent care, or the ER?
2. **What will it cost me?** Nobody publishes a straight answer.

Getting question 1 wrong is expensive in both directions. Under-triage risks
harm. Over-triage costs money you do not have: a retail clinic visit runs about
$60–100, urgent care $150–250, and an emergency room $1,200 or more for the same
complaint. The gap between the cheapest correct answer and the most cautious one
is most of a month's rent.

Getting question 2 wrong is why people skip care entirely.

## What CareFinder does

Four inputs — age, gender, symptoms, ZIP code. No insurance details, because the
users this is built for do not have any.

It returns three things:

| Promise | How it is answered |
|---|---|
| **What level of care** | MedGemma 27B, a medical-domain model, with an emergency gate that runs first |
| **Where to go** | Real facilities from OpenStreetMap, with coordinates, mapped |
| **What it costs** | Live web search per facility, priced for the specific procedures the clinical model identified |

Plus telehealth options, which are geography-independent and often the cheapest
correct answer — particularly in rural areas where the nearest clinic is 40
minutes away.

## The design decision that shapes everything

**A fabricated price is worse than no price.**

For someone deciding whether they can afford to be seen, an invented number is
not a helpful approximation — it is misinformation that changes a health
decision. The same holds for a clinic that does not exist.

So the system never estimates locally. If a price cannot be retrieved, the
interface says *"price not available"*. If no facility is found nearby, it says
so rather than filling the gap. Every displayed figure carries the basis it was
retrieved on.

This sounds obvious. The earlier version of this system did the opposite: it
returned invented prices (`$59–99`) whenever an API key was missing or any
request failed, and synthetic clinics (`"Local Hospital ER"`) whenever a search
came back empty. Neither was distinguishable from real data in the output. See
[docs/DECISIONS.md](docs/DECISIONS.md).

## How it works

```
     age · gender · symptoms · ZIP
                  |
                  v
     ┌────────────────────────────┐
     │  MedGemma 27B — gate       │   Is this a 911 emergency?
     │  (Vertex AI, tool call)    │
     └──────────┬─────────────────┘
       EMERGENCY│  UNKNOWN │  NOT_EMERGENCY
                v          v          │
            call 911   seek care      │   ← a failed check never
                                      │     silently reads as "fine"
                                      v
     ┌────────────────────────────┐
     │  MedGemma 27B — assess     │   care setting, labs, interventions
     └──────────┬─────────────────┘
                v
     ╔══════════════════════════════════════════════╗
     ║  LangGraph orchestrator — Gemini Flash        ║
     ║                                               ║
     ║   find ─> filter ─> price ─> telehealth ─> review
     ║             ^                                │ ║
     ║             └──────── bounded retry ─────────┘ ║
     ╚══════════════════════════════════════════════╝
                v
     care level · mapped places · price bands · online options
```

The emergency gate sits **outside** the graph deliberately. It decides whether
someone calls 911, and that must not depend on the orchestrator being healthy.

Full detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Why an orchestrator was necessary

The first version routed deterministically in Python: the model returned a care
level, and Python decided what to look up. That was deliberate — it kept clinical
judgement in one place and made the system auditable.

It broke on real data.

OpenStreetMap has no reliable "this is an urgent care" tag. `healthcare=urgent_care`
returns **zero** results in both Manhattan NY and Manhattan KS. So a proximity
search returns every medical feature nearby — and a keyword filter cannot tell
a walk-in clinic from a chiropractor, a medspa, or a paediatric practice.

A model can. Given the patient's presentation and the candidate list, Gemini
Flash rejects them with reasons:

> *Pediatric Associates: Pediatric specialty practice that does not treat adult patients.*

That is judgement about the patient's age against the facility's scope. No
keyword list produces it.

The guarantee is preserved in code rather than trust: **the orchestrator can only
keep facilities that were passed to it.** It rejects, it never invents.

## Results

Two full cycles over 20 cases against a live MedGemma 27B endpoint:

- **Zero variance** — 0/20 cases differed between identical runs at temperature 0.1
- **Zero hallucinated facilities, zero hallucinated prices** across 40 case-runs
- **Zero under-triage** — every error was in the cautious direction

And one finding that reshaped the roadmap: the model over-triaged **every**
retail-clinic case to urgent care, which is clinically safe and financially
backwards for the people this is built for.

Full numbers, including what failed: [docs/EVALUATION.md](docs/EVALUATION.md).

## Demo

![CareFinder result for a fever case in Manhattan, Kansas](demo/screenshots/02-urgent-care.png)

*22F, high fever 103F, Manhattan KS. Three real clinics from OpenStreetMap, mapped,
each priced for the labs the clinical model identified — $135–$350 at one, $275–$550
at another 0.6 miles away. Four telehealth options below at $35–$99. Live output,
captured with Playwright.*

More cases, including where the system does badly: [demo/](demo/).

## Design trade-offs

Every row was a real fork, and several were built one way before being replaced.

| Chose | Instead of | Why |
|---|---|---|
| **LangGraph** | LangChain | The review node sends work *back* to pricing. That is a cycle; chains are acyclic. |
| **Pydantic + `response_schema`** | LangChain output parser | The parser asks for JSON and hopes. `response_schema` constrains generation at the API, so malformed output is impossible. |
| **MedGemma 27B** | MedGemma 4B | Clinical reasoning quality. Costs ~54GB of VRAM against ~8GB, which rules out on-device. |
| **Overpass** | Google Places | Free, no key, same OSM data as the map tiles. Costs opening hours and metadata. |
| **Flash orchestrator** | Deterministic Python routing | OSM has no usable "urgent care" tag, so no rule separates a walk-in clinic from a chiropractor. |
| **Flash + search grounding** | Perplexity | One provider, one fewer key, and grounding composes with `response_schema` so pricing returns a typed band. |

Twelve iterations that were built and then replaced — including a local price
table, synthetic facilities, a boolean emergency gate, and a facility search that
returned Bronx clinics for San Jose ZIPs — are recorded in
[docs/TRADEOFFS.md](docs/TRADEOFFS.md).

Three of those bugs shared a shape: **they failed silently and none was findable
offline.** Sampling parameters ignored by the serving container, an empty string
returned for an un-templated prompt, and a schema field dropped because Gemini
cannot express arbitrary-key objects. Each was caught only by running the real
thing and noticing output that was plausible but empty.

## Cost and compute

Measured over one triage — 2 MedGemma calls, ~6 Flash calls.

| | |
|---|---|
| Flash tokens | ~$0.003 |
| **Search grounding ×3** | **~$0.105** |
| Overpass + Nominatim | free |
| **Marginal cost per query** | **~$0.11** |

Grounding is **97%** of it. Token spend is rounding error, so the lever that
matters is how many facilities get priced.

MedGemma is not a per-query cost but a step function — the GPU node bills
~$5–6/hour regardless of traffic, and ~29 minutes of every cold start is
deployment. One query on a cold endpoint effectively costs $3–4; at steady load
with vLLM batching it is ~$0.005. There is no middle.

**Could it run on the edge?** The clinical model at 4B quantised is ~2.5GB and
would run on a phone. The retrieval cannot — facility search and pricing need
live network. The honest design is hybrid, split along the privacy boundary:
inference on device so **symptoms never leave the phone**, retrieval in the cloud
carrying only a ZIP and procedure names. Full analysis in
[docs/TRADEOFFS.md](docs/TRADEOFFS.md).

## How to run it

**Prerequisites:** Python 3.11+, a GCP project with the Vertex AI API enabled and
billing on.

```bash
git clone https://github.com/elakiyasivakumar/carefinder.git
cd carefinder
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**1. Run the tests — needs nothing else.**

```bash
pytest          # 141 tests, no credentials, no endpoint, no network
```

**2. Authenticate.** Use your own login rather than a service-account key file.

```bash
gcloud auth application-default login
cp .env.example .env        # set GOOGLE_CLOUD_PROJECT
```

**3. Deploy MedGemma.** This is the only step that costs money. Model Garden
publishes several configurations; default quota usually allows only this one:

```bash
# medgemma@medgemma-27b-it on g4-standard-48 + 1x NVIDIA_RTX_PRO_6000
# ~29 minutes, and the node bills throughout
gcloud ai endpoints list --region=us-central1     # copy the id into .env
```

**4. Run it.**

```bash
python web/app.py     # http://localhost:8000 — map, prices, telehealth
python cli.py         # terminal
```

**5. Destroy the endpoint when you are done.** It bills while it exists.

```bash
gcloud ai endpoints delete ENDPOINT_ID --region=us-central1
```

### Evaluation

```bash
python run_cycle.py --dry-run          # prints the model-call count, spends nothing
python run_cycle.py --cycles 2         # full pipeline over 20 cases, needs an endpoint
python demo/capture.py                 # Playwright screenshots against localhost
```

`--dry-run` exists because every other command in this section bills.

## Repository

| Path | |
|---|---|
| `triage.py` | Clinical logic: emergency gate, assessment. No cloud SDK, no env at import. |
| `graph.py` | LangGraph orchestration with the bounded retry cycle |
| `orchestrator.py` | Flash nodes: filter, price, telehealth, review |
| `agents/contracts.py` | Pydantic contracts, also used as `response_schema` |
| `services/overpass_service.py` | Facility discovery by proximity |
| `services/gemini_client.py` | Structured, optionally search-grounded Flash calls |
| `audit.py` | Independent hallucination checks over a result |
| `web/` | Flask + Leaflet interface |
| `run_cycle.py` | Full-pipeline evaluation harness |

## Status

A working prototype and an honest one. It is not a product, it has not been
clinically validated, and it is not medical advice.

## License

MIT.
