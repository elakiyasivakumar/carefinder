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

Screenshots from a live run — real model output, real OpenStreetMap facilities,
real retrieved prices: [demo/](demo/).

## Status

A working prototype and an honest one. It is not a product, it has not been
clinically validated, and it is not medical advice.

## License

MIT.
