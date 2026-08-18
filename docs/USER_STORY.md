# The user story

## Who this is for

Someone uninsured or on a high-deductible plan, at home, with a symptom that is
worrying but not obviously an emergency. They are pre-decision, not in crisis.

They are not asking "what disease do I have". They are asking a logistics
question with a price attached: **where do I go, and can I afford it.**

## What they face without this

Three bad options.

**Search the symptom.** Every result escalates. The internet's incentive is to
route you to the most serious possibility, and the answer is always some version
of "see a doctor" with no indication of which kind.

**Call around.** Clinics do not quote prices on the phone. The honest answer they
give is "it depends on what you need", which is true and useless.

**Go to the ER.** It is open, it cannot turn you away, and it is the single most
expensive door in American healthcare. It is where people go precisely because it
is the only option that removes the decision.

That last one is the failure this exists to prevent. An ER visit for something a
retail clinic could treat costs roughly ten times more.

## The journey

**Four fields.** Age, gender, symptoms in plain language, ZIP. No insurance
questions, because the user does not have insurance. No account.

**A safety check first.** Before anything else, MedGemma decides whether this is a
911 situation. If it is, everything else stops — no clinic list, no prices, just
the instruction to call. Cost is not the question when someone is having a heart
attack.

**Then the real answer.** For everything else:

- The level of care actually required, and why
- Named places nearby, on a map, with distances
- A self-pay price band for each, priced for *this* patient's likely procedures
- Online options, which cost less and work anywhere

**And what it could not tell them.** Where a price was not found, it says so.
Where nothing is nearby, it says that too. The user is never left guessing which
parts are real.

## What "good" looks like

A patient with pink eye is told a retail clinic can handle it for about $60–100,
or a video visit for $35 — rather than defaulting to urgent care at $150–250 or an
ER at $1,200.

That single redirection is the entire value of the system. Everything else is in
service of making it trustworthy enough to act on.

## What it deliberately does not do

**Diagnose.** It names likely condition categories internally to reason about
procedures, but does not present a diagnosis to the patient.

**Replace judgement.** It is an informational tool. Every screen says so.

**Handle insurance.** Deductibles and coinsurance are a large, separate problem
and the wrong first problem for this audience.

**Manage chronic conditions.** Acute, episodic complaints only.

## The honest limitation

Evaluation showed the clinical model over-triages: every retail-clinic case in the
test set was escalated to urgent care. Safe, and financially backwards for exactly
the person this is built for.

The system is transparent about this rather than papering over it, because a tool
that always says "urgent care" has stopped answering the question it exists to
answer. See [EVALUATION.md](EVALUATION.md).
