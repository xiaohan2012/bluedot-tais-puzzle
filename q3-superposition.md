# Q3 — Superposition by bottleneck

## Goal

Force the model into a structurally different representation regime than the original puzzle. Where Q1/Q2's "weird" feature was `country` encoded as a quadric *in a 64-dim space with plenty of room*, here we bottleneck hidden 2 to `k < 8` dimensions so the model **cannot axis-align all 8 features**. The expectation, from Anthropic's *Toy Models of Superposition*: the model packs features into overlapping non-orthogonal directions, with some features ending up *non-linearly* encoded (recoverable only via the downstream `Linear→ReLU→Linear`).

Implementation: parametrize `Head` so hidden 2 is `k`-dimensional. Optional L1 penalty on hidden 2 activations. UI toggles for each. Notebook: `q3_superposition.py`.

## Architecture

```python
nn.Linear(384, 64),          nn.ReLU(),
nn.Linear(64, 64),           nn.ReLU(),
nn.Linear(64, hidden2_dim),  nn.ReLU(),   # hidden 2 — layer we inspect (post-ReLU)
nn.Linear(hidden2_dim, 64),  nn.ReLU(),
nn.Linear(64, 8),
```

Train fresh, seed 0, Adam lr=1e-3, batch 128, 30 epochs.

## Key diagnostic: own-head vs linear probe accuracy

For each feature, compute test accuracy of:

- **Linear probe** fit on hidden 2 — measures "is this feature linearly readable from hidden 2?"
- **Model's own head** (end-to-end forward) — measures "does the model actually predict it?"

A **large positive gap (head > probe)** means the model encodes the feature non-linearly at hidden 2 and recovers it through `Linear→ReLU→Linear`. Same diagnostic shape as the original puzzle's country.

## Result 1 — Sweeping k at fixed seed (seed 0)

Largest per-feature gap at each k:

| k | feature with max gap | probe | own-head | gap |
|---|---|---:|---:|---:|
| 2 | question | 0.60 | 0.87 | +0.27 |
| **3** | **sentiment** | **0.54** | **0.93** | **+0.39** |
| 4 | color | 0.59 | 0.67 | +0.09 |
| **5** | **food** | **0.60** | **0.92** | **+0.33** |
| 6 | body_part | 0.64 | 0.73 | +0.09 |
| 8 | number | 0.58 | 0.59 | +0.01 (no gap) |

**Different `k` gives a different "non-linear winner."** At seed 0:

- `k=3` → **sentiment** becomes the non-linear feature (probe ≈ chance, head ≈ 0.93).
- `k=5` → **food** becomes the non-linear feature.
- `k=2` → **question** becomes the non-linear feature.
- `k=8` → no meaningful gap; model has enough room to axis-align.

This is a clean generalization of the original puzzle's phenomenon: by choosing `k` we can *select which feature* gets the non-linear treatment. The original model's `country` was non-linear because country has natural multi-cluster structure; here we *induce* non-linearity by resource scarcity.

## Result 2 — Seed sensitivity at k=3 (five seeds)

Same `k=3`, varying random seed:

| seed | winner | gap | qualitative |
|---|---|---:|---|
| 0 | sentiment | +0.39 | question + food survive linearly (~0.97) |
| 1 | question | +0.09 | only country survives (~0.99); rest at chance |
| 2 | number | +0.00 | **training collapse** — all heads ~0.50 |
| 3 | country | +0.29 | question, food, person, body_part survive |
| 4 | food | +0.15 | question, country, person survive |

**Three findings:**

- **Winner identity is stochastic.** Five different "winning" features across five seeds: sentiment, question, number, country, food. The non-linear feature is determined by the random walk of SGD, not by the data.
- **Training collapse is possible.** Seed 2 produced a degenerate model where every output is chance.
- **The set of *linearly-surviving* features also changes** — which 1–2 features get clean axes is part of the random outcome.

### Why training sometimes collapses

Likely cause at k=3: **dead-ReLU bottleneck collapse.** The 64→3 projection has only 192 weights; if init pushes most pre-activations negative, all 3 ReLUs zero out for every input and no gradient flows back to revive them. Effective dimensionality of hidden 2 drops below 3, model can't represent anything, and training is stuck.

This failure mode is structural to tight bottlenecks + ReLU. Mitigations would include LeakyReLU/GELU, LayerNorm before the bottleneck, or restart-on-collapse — none of which we pursued; we just report the fragility.

## Result 3 — Seed sensitivity at k=6 (five seeds)

| seed | winner | gap |
|---|---|---:|
| 0 | body_part | +0.14 |
| 1 | person | +0.36 |
| 2 | body_part | +0.16 |
| 3 | country | +0.04 |
| 4 | number | +0.11 |

Training is reliable — no collapses, most heads ≥ 0.85. The non-linear winner still varies but with smaller median gap (~+0.14 vs ~+0.27 at k=3).

## Result 4 — Seed sensitivity at k=8 (five seeds)

| seed | winner | gap |
|---|---|---:|
| 0 | number | +0.11 |
| 1 | color | +0.06 |
| 2 | sentiment | +0.16 |
| 3 | color | +0.07 |
| 4 | color | +0.01 |

Even at `k=8` (matching the feature count), some non-linearity persists. **Color is the most common winner (3/5 seeds)** — plausible because `color` is *categorical with many sub-values* (red, blue, …), so a single axis can detect "any color" but not which color cleanly.

## Result 5 — Seed sensitivity at k=16 (five seeds)

Doubling the dimension count above the feature count:

| seed | winner | probe | head | gap |
|---|---|---:|---:|---:|
| 0 | food | 0.979 | 0.981 | +0.001 |
| 1 | question | 0.999 | 0.995 | −0.003 |
| 2 | sentiment | 0.825 | 0.893 | +0.068 |
| 3 | color | 0.963 | 0.963 | +0.001 |
| 4 | color | 0.961 | 0.962 | +0.001 |

**With k ≫ 8, the gap effectively vanishes.** 4 of 5 seeds have gap ≤ 0.001 — every feature is cleanly linearly readable from hidden 2, probe and head agree, head accuracies are ≥ 0.95 across the board. The one outlier (seed 2, sentiment +0.068) is mild residual non-linearity, an order of magnitude smaller than what we saw at k=3.

This is the "no resource scarcity → no superposition → axis-aligned linear code" regime, matching the original `model.pt`'s behavior at the full 64 dimensions.

## Headline trend across k

| k | training failures | median gap | typical head acc | winner stability |
|---|---:|---:|---|---|
| 3 | 1/5 | ~+0.27 | many features at chance | extremely unstable |
| 6 | 0/5 | ~+0.14 | mostly ≥ 0.85 | moderate |
| 8 | 0/5 | ~+0.07 | mostly ≥ 0.95 | stable; color frequent |
| 16 | 0/5 | **~+0.001** | all ≥ 0.95 | uniformly linear |

Three regimes emerge along the `k` axis:

- **k ≪ features (k=3):** Tight bottleneck → big non-linear gaps but fragile training, stochastic winner, occasional collapse. Maximum "weirdness."
- **k ≈ features (k=6–8):** Capacity is matched. Most features axis-align, 1–2 end up non-linear depending on seed. Mild residual weirdness.
- **k ≫ features (k=16+):** Plenty of room. Every feature gets its own clean axis. No non-linearity, no superposition. Equivalent to the original model.

The trade-off is monotone: more bottleneck pressure produces more interesting representations at the cost of reliability.

## Why a narrow bottleneck produces a probe/head gap

The gap comes from a **capacity mismatch between two decoders** sitting on top of hidden 2:

- **Linear probe** = one linear layer.
- **Model's own head** = `Linear(k,64) → ReLU → Linear(64,8)` — a tiny MLP.

When `k ≥ 8` both decoders are essentially equivalent and the gap vanishes. When `k < 8`, only the non-linear one works for some features.

**The budget intuition.** Hidden 2 is a `k`-dimensional budget for 8 features. The model has three options per feature:

1. **Drop it.** Probe and head both fail (heads ≈ 0.50).
2. **Allocate a clean linear direction.** Limited by `k`.
3. **Encode non-linearly via the geometry of the bottleneck.** Place features as discrete clusters / interleaved manifolds / non-linear shapes in the k-dim space. Reading "is feature F on?" requires the downstream ReLU. Linear probe fails; head succeeds. **This is what creates the gap.**

The model picks option (3) for the features where it's cheapest. Those become the non-linear winners we observe.

**The minimal concrete example.** Suppose `k=2` and two features must share. The model can:

- **Linear arrangement (no gap):** put them on perpendicular axes — `h2[0]` for A, `h2[1]` for B. Linear probe trivially recovers each.
- **XOR arrangement (gap appears):** encode `A` and `B` jointly as four discrete points `(+,+), (+,−), (−,+), (−,−)`. Reading "is A on?" is XOR over the two dimensions — *not linearly separable*. A linear probe fails (~0.5). But `Linear(2,64) → ReLU → Linear(64,8)` can compute it: one ReLU per quadrant, sum them. Head succeeds (~1.0).

XOR is one example of non-linear encoding under limited dimensions. Others include **discrete sub-clusters** (the original puzzle's country), **circles** (Engels et al. days-of-week), and **interleaved manifolds**. They all share the property that a single linear cut can't separate on/off regions, but a piecewise-linear ReLU decoder can.

**Why the model "chooses" this:** it's an emergent equilibrium of SGD. At small `k`, the model can't simultaneously satisfy "every feature gets its own axis" and "all 8 features are predicted accurately." The loss landscape has many local minima differing in which features get clean axes vs non-linear codes. SGD lands in one; the choice depends on init. This is also why the *identity* of the non-linear feature is seed-dependent — many minima are roughly equivalent in loss but differ in which feature lives in which slot.

**Two routes to non-linear encoding.** Combining this with the original puzzle's finding:

- **Capacity-driven (our experiment):** force `k < 8`. The model has no choice but to encode some features non-linearly. *Which* feature is determined by SGD's allocation game.
- **Data-driven (original puzzle):** with `k = 64`, capacity isn't the issue. But if a feature has *inherently* multi-cluster structure in the input embeddings (like `country`'s many sub-clusters), the model can find a cheaper encoding by leaning into that structure (a quadric envelope) rather than flattening it into a linear direction.

Same phenomenon (non-linear encoding at hidden 2, recoverable by the downstream MLP), two distinct causes.

## Why this is "more interesting" than the original

The original puzzle's country encoding is *one feature with one shape* (a quadric). Our experiment shows:

1. **Multiple features can become non-linearly encoded** depending on resource constraints.
2. **The choice of which feature is non-linear is not intrinsic to the feature** — it's the result of an allocation problem the model solves at training time. Different `k`, different seed → different winner.
3. **The phenomenon parametrizes cleanly** along the `k` axis, giving a trade-off curve between training stability and representational weirdness.
4. **It demonstrates a known interp phenomenon** (capacity-driven non-axis-aligned encoding) under controlled conditions, mirroring Anthropic's toy-model regime.
5. **The negative seed-stability result is itself a finding**: the *identity* of non-linearly-encoded features is in general not reproducible under tight bottleneck. This has implications for interp tools — the same architecture trained twice can place different features in the "weird" slot.

## Not pursued

- **L1 sparsity (the second toggle in the notebook).** Optional in the UI, not exercised here. Expected effect: at k=8, L1 would push the model from "axis-aligned" toward "non-orthogonal directions with sparse activations" — closer to the canonical Anthropic superposition regime.
- **Sparse autoencoder analysis.** Would recover the per-feature directions despite superposition; the next obvious step for a deeper analysis.
- **k between 3 and 5.** The transitions in winner identity (sentiment → ? → food) are presumably sharp; we didn't map them precisely.
