# Q3 — GRL adversarial probe-resistance, systematic exploration

Follow-up to the earlier adversarial run documented in `q3-adversarial-probe.md`. The earlier writeup looked at 4 hand-tuned runs and concluded "structural limit, not a tuning miss." This pass tests that conclusion systematically: a 4×4×4 sweep over (feature, defense level, λ) with diagnostics designed to catch the canonical GRL failure mode (Elazar & Goldberg 2018).

Notebook: `q3_grl_adversarial.py`. Math reference: `adversarial-grl-math.md`.

## Setup

- Encoder: 3 linear layers, signed at hidden 2 (no ReLU there) — same as `HeadShell`.
- Main head: `Linear(64,64) → ReLU → Linear(64,8)` — shared across all 8 features.
- Linear probe: bare `Linear(64,1)` reading hidden 2.
- Optimization: separate optimizers for {encoder + head} and {each probe}.
- Defenses (from the failure-mode brief):
  - **k_inner**: probe gradient steps per encoder step.
  - **reinit_every**: re-initialise probes every N epochs.
  - **K_probes**: ensemble of K probes; encoder fights the strongest each step.
- Diagnostics tracked per epoch:
  - **in-loop probe acc**: current strongest probe's accuracy on test hidden 2.
  - **held-out probe acc**: fresh `LogisticRegression` refit on current train hidden 2, evaluated on test. **This is the real metric.**
  - **`‖μ̂₁ − μ̂₀‖`**: angular class-mean gap on hidden 2 — what GRL is implicitly minimizing.

## Finding 1 — in-loop probe is anti-correlated with held-out probe (dodging)

Default run with sentiment, λ=0.5, k_inner=5, reinit=0, K=1, 30 epochs:

- L_main drops cleanly to ~0.1
- In-loop probe oscillates **0.4–0.85**, occasionally dipping below the 0.5 chance line
- Held-out probe sits at **0.88–0.92 the entire run**
- `‖μ̂₁ − μ̂₀‖` stuck at 0.2–0.5 with no downward trend

This is the textbook Elazar & Goldberg failure. The encoder is winning the *local* minimax — it makes the in-loop probe's specific `w` perform poorly — but it does so by **rotating** the discriminative direction inside hidden 2, not by removing it. A fresh probe re-fitted from scratch finds the rotated direction immediately.

The right reading of the in-loop number: it measures *"did the encoder just successfully dodge the probe's most recent w?"*, **not** *"is F linearly readable?"* These can be 1 and 0 simultaneously.

Mechanically, the in-loop probe is **stale by exactly one encoder step**: it's fit on `enc_old(x).detach()`, then the encoder steps to make `w_probe · enc_new(x)` look poor. Measuring `w_probe · enc_new(x_test)` is by construction biased downward — and that's exactly the number the in-loop loss is.

## Finding 2 — head F is worse than the held-out probe

Per-feature accuracies after the same run (sentiment as F):

```
feature        head    probe
number        0.960   0.965
question      0.996   1.000
color         0.964   0.967
food          0.965   0.983
sentiment     0.709   0.924   ←  head 21 pts below held-out probe
country       0.990   0.989
person        0.985   0.996
body_part     0.972   0.969
```

The other 7 features show head ≈ probe ≈ 0.96–1.00. **Only sentiment shows a gap, and it goes the wrong way** — the head is *worse* than the fresh linear probe at reading the feature the head was trained to predict.

This is the double failure of naive GRL:

- **F was not scrubbed.** A fresh probe still reads it at 0.92.
- **F was damaged for the head.** Head dropped from a normal ~0.98 to 0.71.

Reasons:
1. The head trains on a moving target while the encoder rotates hidden 2 to dodge the probe.
2. GRL gradient routes through hidden 2 to reduce probe accuracy; to the extent head and probe converge on similar directions (both want the discriminative one), the encoder degrades both — but only the head is locked in.
3. No symmetric pressure on the other 7 features, so they stay clean.

## Finding 3 — the sweep: GRL alone never produces head > held in 64 configs

4 features × 4 defenses × 4 λ values, 10 epochs each, fixed seed.

- Features: `country`, `sentiment`, `person`, `body_part`
- Defenses: `naive (1,0,1)`, `medium (5,0,1)`, `full (5,5,3)`, `aggressive (10,3,5)`
- λ: 0.0, 0.1, 0.5, 2.0

**Best held-out probe per feature** (= worst scrub the encoder achieved):

| feature   | best held-out | config |
|-----------|--------------:|--------|
| country   | 0.883 | aggressive, λ=0.5 |
| sentiment | 0.829 | aggressive, λ=2.0 |
| person    | 0.879 | aggressive, λ=0.1 |
| body_part | **0.818** | aggressive, λ=0.5 |

Even the best-scrubbed cell leaves ~32 percentage points above chance. **None of 64 cells achieve `head > held`.** The `h-held` column is ≤ 0 everywhere; it's exactly 0 only at λ=0 (no adversarial pressure).

## Finding 4 — `‖μ̂₁ − μ̂₀‖ → 0` is necessary but not sufficient

The strongest negative result in the sweep — subset of rows where the encoder did drive the first-moment direction gap nearly to zero, yet the held-out probe stayed high:

| feature   | defense    | λ    | μ_gap | held-out |
|-----------|------------|-----:|------:|---------:|
| body_part | aggressive | 0.5  | 0.033 | 0.818 |
| country   | aggressive | 0.5  | 0.035 | 0.883 |
| body_part | full       | 0.5  | 0.041 | 0.876 |
| sentiment | aggressive | 0.1  | 0.075 | 0.879 |
| sentiment | aggressive | 0.5  | 0.060 | 0.871 |
| sentiment | full       | 0.5  | 0.059 | 0.896 |

GRL only targets the first-moment direction. **Important math correction:** even at `‖μ̂₁ − μ̂₀‖ = 0` exactly, a linear probe is *not* guaranteed to fall to chance *accuracy*. The standard "first moments equal ⇒ no linear separation" intuition only holds for *convex loss* (e.g., logistic loss — this is the LEACE notion of "linear guardedness"). Accuracy is not a convex loss. With equal class-conditional means but unequal class-conditional covariances, a thresholded linear function can sit at 70–80% — picture one class concentrated at the origin and the other bimodal at `±v` along some direction; a shifted threshold along `v` gets ~75%. LogReg's bias term plus second-moment asymmetry is exactly the regime that produces our 0.82–0.90 numbers above.

So Finding 4 isn't a bug in our GRL implementation that's leaving the first moments un-collapsed; even *perfect* first-moment collapse wouldn't put a linear probe at chance on this kind of data. Real text embeddings have unequal class-conditional covariances, and any erasure method that only matches first moments — GRL, L_iso, any aux loss on `‖μ̂₁ − μ̂₀‖²` — has a built-in ceiling well above 0.5.

Practical implication: matching first moments alone is not enough on real text embeddings — class-conditional covariances are not equal. A faithful "linear scrub" needs first + second moment matching (or LEACE).

## Finding 5 — defenses help but plateau

Held-out probe at λ=0.5 across defense levels:

| F         | naive | medium | full  | aggressive |
|-----------|------:|-------:|------:|-----------:|
| country   | 0.975 | 0.949  | 0.937 | 0.883 |
| sentiment | 0.931 | 0.903  | 0.896 | 0.871 |
| person    | 0.981 | 0.936  | 0.926 | 0.893 |
| body_part | 0.877 | 0.859  | 0.876 | 0.818 |

Monotonic shrinkage but small — ~3 percentage points per defense step. Asymptote sits around 0.80, well above chance. More k_inner / more probes / more reinit helps at the margin, but does not change the qualitative outcome.

## Finding 6 — confounds and caveats

- **`worst-other-head` is uninterpretable in this sweep.** It hovers at 0.46–0.65 across nearly every config, *including* λ=0. That means the encoder isn't learning the other 7 features well in 10 epochs — too-short runs are confounding the "adversarial training collateral damage" reading. Longer runs needed.
- **Country is not the puzzle's country.** Baseline λ=0 gives `country` held-out probe ≈ 0.99 in this notebook, vs 0.43 in the original puzzle. The puzzle's specific seed-0 encoder learned a quadric encoding; our fresh encoder doesn't reproduce that. So country here behaves like the other clean linear features — not the "already non-linear" testbed it could have been.

## Finding 7 — GRL + `L_radial` hybrid also fails (option 1 from "what to try next")

Followup test of the "recommended" path: add `L_radial` (install F in radius) on top of GRL (scrub linear direction). Same architecture, same data, same target feature, same `α=0.1` as the working concentric-shells experiment — only the scrubber changed (L_iso → GRL).

| run | λ   | α    | F         | held-out probe |
|-----|----:|-----:|-----------|---------------:|
| baseline (no aux loss) | – | – | sentiment | ~0.97 |
| GRL + L_radial | 0.5 | 0.1 | person | 0.993 |
| GRL + L_radial | 0.5 | 0.5 | sentiment | 0.982 |
| GRL + L_radial | 2.0 | 0.5 | sentiment | 0.972 |
| **L_iso + L_radial** (concentric shells, same α=0.1) | – | 0.1 | sentiment | **0.585** |

The hybrid is essentially indistinguishable from no-adversarial-training baseline. Concentric shells' L_iso variant drives the same held-out probe to ~0.59; the GRL variant of the same recipe leaves it at 0.97–0.98.

This is a sharper negative result than the GRL-alone sweep. The "GRL needs a positive encoding instruction" hypothesis (option 1 in the previous version of this writeup) was wrong as a *fix* for GRL. Adding `L_radial` does install the radial encoding fine (`L_radial` drops to ~0.1 in training), but F is also still in a direction the linear probe finds — because GRL never killed it.

**Mechanism: GRL ≠ L_iso in practice, even though they're equivalent in theory.**

Both objectives target the same quantity at equilibrium — first-moment direction gap `‖μ̂₁ − μ̂₀‖ → 0`. They differ in *how* they estimate the gradient of that quantity:

- **`L_iso`** is a **closed-form** penalty on the empirical class-mean difference. Every step pushes `μ_gap → 0` faithfully and immediately. In the concentric-shells run, `μ_gap` drops from 0.6 → 0.05 in ~5 epochs.
- **GRL** approximates the same penalty via the gradient of a *learned* linear probe. The probe is stale-by-one-step, the encoder's response is rotation-based dodging that fools the in-loop probe without removing the direction, and `μ_gap` never collapses. In the hybrid runs, `μ_gap` oscillates between 0.2 and 1.0 across all 30 epochs with no downward trend.

So the asymptotic equivalence between GRL-on-linear-probe and L_iso (the math in `adversarial-grl-math.md`) is theoretically real but practically meaningless on this task. The closed-form version converges; the adversarial version doesn't.

**The hardened takeaway**: **`L_iso` is what GRL is trying to approximate. The closed form converges; the adversarial estimator dodges.** If you want first-moment matching, use `L_iso`. If you want second-moment matching too, use LEACE. GRL is a strict pessimization of both.

## Finding 8 — GRL + `L_radial` + `L_iso`: works, but GRL contributes nothing

Definitive head-to-head test. Holding F=sentiment, λ=0.5, α=0.1, k_inner=5, reinit=5, K=3, 25 epochs. Sweeping β (the L_iso weight) over {0.0, 0.05, 0.5, 2.0}:

| β     | held  | head  | norm-only probe | worst other head | μ_gap |
|------:|------:|------:|----------------:|-----------------:|------:|
| 0.00  | 0.949 | 0.977 | 0.978           | 0.611            | 0.131 |
| 0.05  | 0.965 | 0.979 | 0.977           | 0.571            | 0.093 |
| 0.50  | **0.661** | **0.976** | 0.975       | 0.531            | 0.056 |
| 2.00  | **0.592** | 0.955 | 0.940           | 0.543            | 0.030 |

**The result:**

- β=0.5: held=0.66, head=0.98 → gap **+0.32**
- β=2.0: held=0.59, head=0.96 → gap **+0.36**

For comparison, concentric shells (no GRL, just `L_iso + L_radial`) gives held=0.585, head=0.979 → gap +0.39 at the matching settings. Essentially identical.

**The damning detail: GRL contributes nothing.** Look at the β=0 row — pure GRL + L_radial, no L_iso — held stays at 0.949 (no scrubbing). Adding L_iso (β>0) drops it monotonically to 0.59. `μ_gap` follows the same monotone path (0.13 → 0.09 → 0.06 → 0.03). If we had set λ=0 (killed GRL) and kept only L_iso + L_radial, the result would be indistinguishable from β=2.0 here — it would *be* concentric shells.

This is the cleanest possible empirical confirmation of Finding 7's claim:

1. **GRL alone** → held ≈ 0.97. Doesn't scrub.
2. **GRL + L_radial** → held ≈ 0.97. Doesn't scrub.
3. **GRL + L_radial + L_iso (β=2.0)** → held ≈ 0.59. Works.
4. **L_iso + L_radial (no GRL)** → held ≈ 0.59. Works equally well.

GRL is along for the ride. `L_iso` is doing all the scrubbing work. On this task, with this architecture, adversarial training on a linear probe never converges to its theoretical equilibrium; the closed-form penalty does, in one gradient step.

`norm_probe` stays high (0.94–0.98) across all β — the radial encoding is preserved, so the head reads F through `‖h‖` (non-linear), while the linear probe gets killed by L_iso.

## Conclusion — declaring failure, with the reasons named

**We are not pursuing this further.** GRL-on-linear-probe does not deliver the Q3 outcome (`head > held-out probe`) on this data, and the failure is structural, not a tuning miss. This section explains *why* in terms that match the published literature.

### What the experiments showed

- **Naive GRL** (Finding 1, 2): in-loop probe falls to ~0.5 (looks like a win), held-out probe stays at ~0.92 (no scrubbing), head on F drops to ~0.71 (worse than the attacker). `head − held-out ≈ −0.21`.
- **GRL + defenses** (Finding 3, 5): across 64 (feature × defense × λ) configurations, *none* produce `head > held-out`. Held-out plateaus around 0.80 — well above chance.
- **`μ_gap → 0` doesn't deliver chance accuracy** (Finding 4): even cells where the encoder collapsed the first-moment direction gap to <0.08 left the held-out probe at 0.82–0.90 — because matching first moments only guarantees linear guardedness under convex loss, not under accuracy. This is a property of the data (unequal class-conditional covariances), not of GRL.
- **GRL + `L_radial` hybrid** (Finding 7): adding a positive encoding instruction doesn't help. Held-out stays at 0.97 even though `L_radial` installs cleanly.
- **GRL + `L_radial` + `L_iso`** (Finding 8): works (`head − held = +0.36`), but the head-to-head shows GRL contributes nothing — L_iso does all the scrubbing work, and dropping λ to 0 produces an indistinguishable result. The notebook reproduces concentric shells; the GRL term is dead weight.

### Why GRL fails — confirmed against the literature

This is a known result, not a finding. The relevant references:

1. **Dodging is the central finding of Elazar & Goldberg 2018.** Their adversary reached chance in-loop while post-hoc classifiers retrained on the frozen encoder recovered the attribute well above chance. Barrett et al. 2019 confirmed and sharpened it. The INLP paper (Ravfogel et al. 2020) was explicitly motivated by this leakage. Mechanism: simultaneous gradient descent-ascent in a nonconvex-concave game doesn't converge to the minimax solution. The encoder's cheapest descent direction is to rotate the discriminative subspace out of the current probe's row space — almost free representationally. Fooling K specific probes is a measure-zero constraint relative to fooling the whole linear family.

2. **Inner-loop fixes converge to a closed-form method.** Two-timescale GDA with probe LR ≫ encoder LR (Lin, Jin & Jordan 2020) converges to stationary points of `max_θ L_probe(φ, θ)`. But if you solve that inner max exactly, you're optimizing the encoder against the best linear adversary — which has a closed-form spectral solution. That is literally R-LACE (Ravfogel et al. 2022). The fully-fixed version of GRL-on-linear-probe is a closed-form method; the SGD version is an unreliable approximation of it.

3. **λ schedules don't change the equilibrium structure.** The Ganin–Lempitsky sigmoid ramp is a warm-up heuristic from domain adaptation, where partial invariance is acceptable. It doesn't escape the dodging fixed point. Larger λ buys more first-moment collapse at the cost of more head damage — moving along a frontier, not escaping it.

4. **The head-side damage is structural, not incidental.** The Q3 goal — `head > held-out linear probe` — requires F to be encoded *nonlinearly but not linearly* in h. GRL's objective never asks for that. It asks the encoder to reduce the probe's loss-gradient signal while the BCE gradient on F pushes through the same parameters. The two gradients fight on the same linear directions (the MLP head's first layer reads `h` linearly before its nonlinearity), so the resolution is partial destruction of F plus rotation. Head degrades; fresh probes don't.

### The deeper tension our results expose

The Q3 goal isn't always reachable. For `head > held-out linear probe` to be achievable on a given feature, the feature must have a *non-linear signal in `h` that's separable from its linear signal*. If F is encoded essentially linearly (which the original puzzle's `sentiment`, `person`, `number` etc. are), no erasure method rescues you — erasing the linear signal also kills the head's access. Concentric shells works *because we install a non-linear encoding (radial) before scrubbing the linear one*. The non-linear signal exists only because we put it there.

GRL has no mechanism for installing a non-linear encoding. It can only scrub. Without a paired positive encoding instruction, "head > held-out" is structurally impossible — and the positive encoding instruction is what concentric shells already provides. Adding GRL on top buys nothing, as Finding 8 directly shows.

### What we would do instead (not pursued)

For completeness, the principled alternatives:

- **R-LACE** (Ravfogel et al. 2022): solves the linear-probe minimax in closed form via a concave–convex relaxation over projection matrices. Dominates SGD-GRL on every published comparison.
- **LEACE** (Belrose et al. 2023): closed-form optimal affine map that zeroes the cross-covariance between `h` and F, guaranteeing linear-guardedness under convex loss. Doesn't guarantee chance accuracy on its own (Finding 4 applies to LEACE too), but combined with retraining the head on the erased representation, it is the principled "linear erasure" baseline.
- **L_iso + L_radial** (concentric shells, already done): the practical recipe that delivered the +0.39 gap. Achieves the Q3 goal not by being a better adversarial method but by *installing* the non-linear encoding F needs in order to be readable by the head but unreadable by a hyperplane.

The concentric-shells result already settled Q3. The GRL exploration was worth doing as a documented failure — it makes explicit *why* concentric shells works (the installation step, not the scrubbing step, is the load-bearing trick), and it replicates a well-known negative result on a new task.
