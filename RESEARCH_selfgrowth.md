# Research: can a model self-grow with NO scaffolding?

## The precise problem
"Scaffolding" = external, non-learned machinery that (a) DECIDES to grow, (b) ALLOCATES new parameters,
(c) SPLICES them in, (d) resets the optimizer. Zero-scaffolding self-growth requires all of this to be either
unnecessary or part of the model's own intrinsic (learned / forward-pass) dynamics.

## The impossibility core (be honest first)
A **fixed set of weights cannot allocate new weights to itself from inside a pure forward pass.** Instantiating a
new parameter tensor is a discrete structural act that some process *outside* the differentiable computation must
perform. This is true of every growing-net method (Net2Net, gradual stacking, dynamic sparse training): the
*allocation* is always a harness. So "grow = physically add parameters, with zero external code" is **impossible**
in the strict sense.

**Therefore the useful question is a REFRAME:** don't define growth as "add parameters." Define it as *"the model
deploys more of its capacity/computation when its own state says the task demands it."* Under that definition,
scaffolding-free self-growth **is** achievable. Three architectures do it, ordered by how cleanly:

---

## Approach A -- ADAPTIVE COMPUTATION / HALTING  (recommended; best fit for our recurrent core)
**Idea:** one weight-tied operator (a single block) applied a *learned number of times* per input. The model has
its own **halting unit** that, from its own activations, decides "I've thought enough" (Adaptive Computation Time,
Graves 2016; PonderNet; Universal Transformer; Deep Equilibrium nets = fixed-point iteration to self-chosen depth).
- **Growth reframed as:** effective DEPTH is a per-input, in-forward-pass decision the model makes itself.
- **Scaffolding: NONE.** No allocation (weights are tied/fixed), no splice, no external trigger. The halting neuron
  is trained end-to-end; deciding to "go deeper" is literally the model's own output.
- **Why it fits US:** our core is already a recurrent scan. Make the block weight-tied + add a learned halt ->
  the brain *thinks longer on harder inputs, by its own choice.* Capacity-on-demand, fully intrinsic.
- **Limit:** grows COMPUTE/effective-depth, not parameter COUNT. But that IS the honest form of "use more brain
  when needed." Ponder cost is bounded by a learned prior (PonderNet) so it doesn't runaway.

## Approach B -- GATED OVER-PROVISIONED CAPACITY (learned recruitment)
**Idea:** pre-allocate a large pool of units/experts that start DORMANT (gated off); the model's own learned gates
RECRUIT them as tasks demand (sparsely-gated MoE with learned routing; PathNet; lottery-ticket masking).
- **Growth reframed as:** "physically fixed, functionally growing" -- the model turns on latent capacity it
  already has, by gradient-driven gates.
- **Scaffolding: NONE for the decision** (gates are learned) -- but capacity has a pre-set CEILING and dormant
  units still cost memory. So it's bounded growth, not open-ended.

## Approach C -- SELF-GENERATED WEIGHTS (fast-weight programmer / hypernetwork)
**Idea:** the model outputs the parameters of new units from its own state (Schmidhuber fast-weight programmers;
hypernetworks). "Growth" = the model generating more of its own weights as a function of what it has seen.
- **Scaffolding:** decision AND parameter VALUES are fully the model's; only the *instantiation* of the generated
  tensor into a running module is external (and mechanical). Closest to "the model literally grows itself."
- **Limit:** the generator is fixed-size; hard to train stably; the runtime splice remains.

## Approaches that do NOT dissolve the scaffolding (for completeness)
- Loss-plateau triggers (what we have now): the DECISION is a hand-coded rule on an EXTERNAL signal (label loss).
  Scaffolding.
- Dynamic sparse training (RigL/SET): grow/prune connections by gradient magnitude -- but on an external
  "every-N-steps top-k" schedule. Partially scaffolded.
- Net2Net / stacking: allocation + schedule both external. Fully scaffolded (this is our old grow.py).

---

## Synthesis / recommendation
1. **Strict "add params with zero external code" is impossible** -- weights can't malloc weights. Say this plainly.
2. **The achievable, honest goal is capacity-ON-DEMAND decided by the model's own signal.** Best realization for
   this project = **Approach A (adaptive halting) on our recurrent core**: weight-tied block + a learned halting
   unit -> the brain chooses how deep to think, per input, entirely inside its forward pass. No allocator, no
   trigger heuristic, no splice. That is genuinely scaffolding-free "self-growth" (in the compute/depth axis).
3. **For parameter growth specifically** (if we insist on more weights): the *decision* can still be internalized
   by triggering on the model's OWN uncertainty/surprise signal (not external loss); the allocation stays a
   necessary developmental controller. Frame it as METABOLISM, not computation -- honest and defensible.
4. **Combine:** adaptive-halting (A) for intrinsic depth-on-demand + surprise-triggered param-growth (metabolic
   controller) for long-horizon capacity. A gives true zero-scaffolding capacity elasticity now; the metabolic
   controller handles the irreducible allocation honestly.

## Concrete next experiment (small, testable)
Make our block **weight-tied + PonderNet halting**: train on a task with variable difficulty; show the model
spends MORE compute-steps on harder inputs BY ITS OWN halting choice (halt distribution correlates with
difficulty), with NO external step-count schedule. That is scaffolding-free self-scaling of effective depth,
provable small -- the honest core of "the model grows itself."
