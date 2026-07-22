# Ensemble consensus & per-configuration uncertainty

**Status:** proposal / idea — not yet an accepted decision
**Date:** 2026-07-22
**Author:** Juan (idea), captured with Claude
**Related:** [ADR 0001 — per-MLIP envs](../adr/0001-per-mlip-envs.md), `CONTEXT.md`

## The idea in one line

Run a configuration through a *cluster* of machine-learning potentials instead
of a single one, so a run reports not just a value but a **consensus value plus
a spread** — an uncertainty estimate for that configuration.

## Why this matters

A single MLIP gives a point estimate with no error bar. For a lot of the work
this package supports — adsorption energies, transition-state barriers, picking
which of several candidate configurations is physically meaningful — the
question that actually matters is *how much can I trust this number*. A
committee of potentials answers that: where the models agree, the prediction is
well-constrained; where they disagree, the configuration sits in a region the
potentials extrapolate into, and the result deserves suspicion (or a DFT
check).

This is the established **query-by-committee** idea from the MLIP
active-learning literature, applied as a first-class run mode rather than a
bespoke script.

## The dominating constraint (read this first)

**MLIP packages do not coexist in one Python environment.** Per
[ADR 0001](../adr/0001-per-mlip-envs.md), `mace-torch`, `fairchem-core`,
`sevenn`, and `chgnet` pin mutually incompatible `torch` / `torch_geometric` /
`e3nn` versions. `CONTEXT.md` even records the exact question this proposal
raises —

> *"I want to compare MACE and UMA on the same structure — can I install both?"*
> *"Not in one env."*

So the design **must** split into two regimes:

| Regime | Members | Where they run | Feasibility |
|--------|---------|----------------|-------------|
| **Same-package committee** | UMA task-heads (`omat`/`oc20`/`oc22`/`oc25`), or several MACE models | one MLIP env, one process, in-memory | buildable now |
| **Cross-package committee** | MACE + UMA + CHGNet | one env *per* member, orchestrated | needs the deferred subprocess bridge |

ADR 0001's own "Deferred" section names the trigger for revisiting the
cross-package architecture: *"Revisit only if users start asking for 'compare
MACE and UMA in one run' workflows."* This proposal **is** that ask. The plan
below reaches it in phase 2 rather than paying for it up front.

## Scope decisions (settled)

These were chosen deliberately; alternatives and their trade-offs are recorded
so a future reader knows they were considered, not overlooked.

1. **Ensemble composition — same-package first, cross-package later.**
   Phase 1 ensembles only members that share one env. Honest about ADR 0001
   while still reaching the end goal.
   *Caveat carried forward:* members that share a backbone (UMA task-heads) are
   **correlated**, so their spread is a *lower bound* on true uncertainty. The
   committee flags disagreement reliably but should not be read as a calibrated
   error bar until cross-package (phase 2) or genuinely independent models are
   in the mix. This caveat must be surfaced in the output, not buried.

2. **Uncertainty source — live committee disagreement during the run.**
   One trajectory, driven by **committee-averaged forces**; at every optimizer
   step all members evaluate the current geometry, the mean force takes the
   step, and the **spread of forces across members** is recorded as the live
   uncertainty signal. This isolates model disagreement at each geometry the run
   actually visits.
   *Alternatives considered:* (a) *independent runs* — each model relaxes on its
   own, compare the N results; rejected as the primary mode because it mixes
   model uncertainty with different minima and is not the configuration's
   uncertainty. (b) *fixed-geometry single-point committee* — relax with one
   driver, then evaluate all members on the final structure; simpler but blind
   to disagreement *along the path*. Both remain useful as reporting options
   (see Open questions).

3. **Quantity carrying the uncertainty — forces and relative energies.**
   Force disagreement is the primary signal; relative energies (barriers,
   adsorption/reaction energies) are the comparable secondary signal.
   *Explicitly not* absolute per-configuration energy across heterogeneous
   members: different model families sit on different energy zeros (different
   reference energies, different DFT functionals), so their absolute-energy
   spread is dominated by constant offsets, not physics. Within a same-package
   committee absolute energy *is* comparable, so it may be reported there with a
   clear "same-reference only" caveat.

4. **First command — `optimize`.**
   Most tractable: relax, and report the force-disagreement trace plus the
   energy/force spread at the minimum as a clean per-configuration UQ number.
   MD (committee-force UQ per frame) and NEB (consensus barrier + variance)
   follow, reusing the same consensus layer.

## How it works — `optimize` (phase 1)

Committee-driven relaxation, all members in one env:

1. Load N calculators from one MLIP env (e.g. the four UMA task-heads, or K
   MACE models). Loaded once, reused across steps — the existing `batch`
   model-reuse pattern generalizes here.
2. At each optimizer step, on the current geometry:
   - every member computes forces (and energy),
   - the **mean force** drives the ASE optimizer step,
   - record per-step: mean energy, energy std, and a **force-disagreement
     scalar** — e.g. the per-atom force-vector standard deviation across
     members, reduced to its max and mean over atoms.
3. On convergence, the outputs are:
   - the consensus relaxed structure (from the mean-force trajectory),
   - a **disagreement trace** over the relaxation (CSV, alongside the existing
     `opt_convergence.csv`),
   - per-configuration UQ summary: peak and final force-disagreement, final
     energy spread, and the list of committee members.

The uncertainty summary is a natural new block in the existing
`mliprun_run.json` run record — it already has a `provenance`/`stages` schema,
and a committee run is exactly a run whose provenance is "N models, these tags"
and whose outcome carries a spread. This reuses infrastructure rather than
inventing a parallel output.

## Phased plan

- **Phase 1 — same-package `optimize`.** Committee-averaged-force relaxation,
  force-disagreement trace, UQ summary in the run record. Single env, in-process.
  Delivers the core value with no new architecture.
- **Phase 2 — MD and NEB.** Reuse the consensus layer: MD reports committee
  force disagreement per frame (the classic extrapolation detector); NEB reports
  a consensus barrier with per-image force disagreement and a barrier spread.
- **Phase 3 — cross-package orchestration.** The ADR-0001-deferred subprocess
  bridge: each member runs in its own MLIP env, a driver process gathers forces
  per step and aggregates. Unlocks genuinely heterogeneous committees (MACE +
  UMA + CHGNet) and turns the correlated-members caveat from phase 1 into a real
  error bar. This is the multi-month piece; phases 1–2 must earn it first.

## What "done" looks like (phase 1)

`mlip optimize` (or a sibling command / `--ensemble` flag — see open questions)
accepts a set of same-env MLIP members, relaxes a structure under
committee-averaged forces, and writes: the relaxed structure, a per-step
disagreement trace, and a per-configuration uncertainty summary in
`mliprun_run.json` — with the correlated-members caveat stated in the output.

## Open questions (for a later brainstorming/spec pass)

- **Interface.** New `--ensemble "uma-omat,uma-oc20,..."` on the existing
  commands, or a distinct `ensemble` sub-tool? The former reuses all the option
  plumbing; the latter keeps single-model runs clean.
- **Disagreement metric.** Per-atom force-vector std reduced how — max over
  atoms, mean, RMS? Report the full per-atom field or just the scalar summary?
  (A per-atom field would let you see *which* atoms the models disagree on — often
  the adsorbate or the reacting bond.)
- **Driving force.** Mean force (active committee, chosen) vs. drive-with-one-
  model-observe-the-rest (passive committee — cheaper reasoning about "what one
  model would have done", but the trajectory is then that one model's). Worth a
  flag?
- **Reporting mode.** Alongside the live trace, also offer the cheap
  fixed-geometry single-point committee on the final structure, for users who
  only want an error bar on the endpoint?
- **Cost.** N forward passes per step is N× the compute. Acceptable for
  optimize; for long MD it may need a stride (committee every M steps).
- **Weighting / consensus rule.** Plain mean, or weighted (e.g. by a per-model
  reliability)? Plain mean for v1; leave the hook.

## Next step

If this direction is right, the natural follow-up is a `superpowers:brainstorming`
pass on **phase 1 only** (same-package `optimize`), producing a design spec and
then an implementation plan — the same path the run-record feature took.
