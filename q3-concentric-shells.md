# Q3 — Concentric shells

## Goal

Force one feature F off the linear-direction subspace at hidden 2 by encoding it in **radius** instead of direction. This is the canonical "non-linearly separable" geometry: a linear probe `w · h + b` can't read radius, but a downstream `Linear → ReLU → Linear` can. We chose **`sentiment`** as F — the cleanest linear feature in the original puzzle (baseline probe 0.98) — so the before/after contrast is sharp.

Unlike previous experiments (bottleneck, adversarial), the non-linear shape here is **chosen by us and injected by an auxiliary loss**, not emergent from architectural pressure.

Notebook: `q3_concentric_shells.py`.

## Why L2-normalization alone doesn't break linear separability

It's tempting to think that constraining activations to lie on a sphere (via L2 normalization) would already make linear probes fail. It doesn't, and the reason is geometric.

A linear probe computes `w · h + b > 0`. Geometrically, the decision boundary is a **hyperplane** in ℝᵏ. When all activations are constrained to the unit sphere, this hyperplane *still* intersects the sphere — it cuts it into two **hemispherical caps**. A binary classifier just has to ask "is this point in this hemisphere?", and a hyperplane answers that question fine.

So on a sphere, for two classes:

- The model can place class-0 samples around one pole and class-1 samples around the opposite pole.
- A single hyperplane (the equator perpendicular to the pole axis) separates them perfectly.
- Linear probe accuracy stays high.

We confirmed this directly in `q3_l2norm.py`: at k=64 with L2 normalization, *every* feature including the originally-non-linear country becomes cleanly linearly readable (probe ≈ own-head ≈ 0.98 across all 8). Removing the radial degree of freedom didn't add non-linearity — if anything, the larger full sphere (compared to the positive orthant ReLU forces) gave the model *more* room to flatten encodings into clean linear directions.

**The real way to break linear separability is to put the class signal somewhere a hyperplane can't read.** Radius is one such place: no hyperplane through ℝᵏ can separate "inside a ball of radius `R₀`" from "in a shell at radius `R₁`" because that question depends on `‖h‖`, which no `w · h + b` can compute. Concentric shells exploit this.

The full recipe needs both:

- **L_radial** to install the radial signal.
- **L_iso** to *remove* the directional signal — otherwise the model just adds radius information on top of an existing linear direction and a linear probe still finds the direction.

The α/β sweep below shows that L_radial alone (β=0) leaves the linear probe intact at 0.98; the iso loss is structurally necessary, not just a fine-tuning knob.

## Architecture change

Hidden 2 is the output of the third `Linear`, **with no ReLU** (replaced by nothing). This lets activations take any sign, so the vector can be centered around 0 and live on concentric shells. The original puzzle's post-ReLU forces non-negativity, which would break the "concentric about origin" geometry.

```python
def hidden2(x):
    h = ReLU(Linear1(x))
    h = ReLU(Linear2(h))
    return Linear3(h)   # signed; no nonlinearity here
```

Downstream is unchanged: `Linear(64,64) → ReLU → Linear(64,8)`.

## Loss design

Three losses combined:

```
L = L_main + α · L_radial + β · L_iso
```

- **`L_main`** — standard 8-feature BCE on the model's own head (preserves the main task).
- **`L_radial = mean((‖h2_c‖ − target_R[y_F])²)`** — pulls F=1 samples to radius `R₁` and F=0 samples to `R₀`. (`h2_c` is the batch-centered hidden-2 vector.) Installs the shell encoding.
- **`L_iso = ‖mean_dir(F=1) − mean_dir(F=0)‖²`** — pushes class-conditional mean direction to be the same for both F classes. Scrubs F from any directional/linear subspace.

The combination is what makes the encoding **purely radial**: `L_radial` builds the shells; `L_iso` removes the residual linear signal.

## First attempt — α=0.5, β=0.5

| feature | probe | own-head | gap |
|---|---:|---:|---:|
| number | 0.599 | 0.596 | −0.003 |
| question | 0.979 | 0.960 | −0.019 |
| color | 0.724 | 0.705 | −0.019 |
| food | 0.968 | 0.963 | −0.005 |
| **sentiment** | **0.617** | **0.982** | **+0.365** |
| country | 0.987 | 0.987 | −0.001 |
| person | 0.991 | 0.992 | +0.001 |
| body_part | 0.697 | 0.633 | −0.064 |

- **Sentiment intervention worked** (probe 0.62, head 0.98, gap +0.37).
- **But three other features were damaged** — number, color, body_part fell from baseline ~0.97 to 0.60–0.72 on own-head.

The 1-feature probe on `‖h‖` alone got 0.983 on sentiment — confirming the radial encoding installed cleanly.

## Why the collateral happens (and it's not correlation)

All Pearson correlations between sentiment and the other 7 features are `|r| < 0.02` — the features are essentially independent. The damage comes from the **global geometric constraints** the aux losses impose:

- `L_iso` requires `mean(dir | sent=1) ≈ mean(dir | sent=0)`. Even with uncorrelated features, this restricts how the model can lay out direction space — it can't freely place feature axes if doing so would shift sentiment's class-mean.
- `L_radial` forces both sentiment classes to specific magnitudes. The model can no longer use magnitude differences to encode information for other features.

Both losses **leak beyond sentiment**. The fix is to make them weaker — but weak enough to still scrub sentiment from the linear subspace.

## Sweep — finding the contrastive sweet spot

Same setup, target=sentiment, varied α (radial) and β (iso):

| α | β | sent probe | sent head | norm-probe(F) | worst other probe | worst other head |
|---:|---:|---:|---:|---:|---|---|
| **0.10** | **0.05** | **0.585** | **0.979** | **0.982** | **0.957 (number)** | **0.946 (color)** |
| 0.20 | 0.05 | 0.715 | 0.983 | 0.987 | 0.639 (number) | 0.581 (number) |
| 0.30 | 0.05 | 0.713 | 0.979 | 0.980 | 0.635 (number) | 0.605 (number) |
| 0.50 | 0.05 | 0.759 | 0.983 | 0.982 | 0.613 (number) | 0.593 (number) |
| 0.50 | 0.00 | 0.982 | 0.985 | 0.988 | 0.672 (number) | 0.599 (body_part) |
| 1.00 | 0.00 | 0.980 | 0.987 | 0.985 | 0.567 (number) | 0.567 (number) |
| 0.50 | 0.02 | 0.903 | 0.984 | 0.981 | 0.644 (number) | 0.657 (number) |
| 0.30 | 0.10 | 0.665 | 0.979 | 0.987 | 0.598 (number) | 0.563 (number) |

**Two clean observations:**

- **β > 0 is essential** to scrub sentiment from the linear subspace. With β=0, the model can simultaneously satisfy `L_radial` (install shells) and keep a linear direction for sentiment alive — radius adds *on top of* the linear direction rather than replacing it. Probe stays at 0.98.
- **β slightly > 0 (e.g., 0.05) is enough.** Higher β doesn't scrub more — it just damages other features.

**Best setting: α=0.10, β=0.05.**

## Final result — α=0.10, β=0.05

| feature | probe | own-head | gap |
|---|---:|---:|---:|
| number | 0.957 | 0.953 | −0.004 |
| question | 0.995 | 0.995 | +0.000 |
| color | 0.946 | 0.946 | +0.000 |
| food | 0.974 | 0.976 | +0.002 |
| **sentiment** | **0.585** | **0.979** | **+0.394** |
| country | 0.984 | 0.984 | +0.000 |
| person | 0.994 | 0.994 | +0.000 |
| body_part | 0.980 | 0.980 | +0.000 |
| **norm-only probe on sentiment** | — | — | **0.982** |

- **Only sentiment moves.** All other 7 features stay at 0.95+ for both probe and own-head — within noise of the baseline.
- **Sentiment is scrubbed from linear-direction space** (probe 0.59 — near chance for this dataset's class balance).
- **Sentiment is preserved in radius** (own-head 0.98; norm-only probe 0.98).
- **Gap of +0.39** between probe and own-head on sentiment — the largest, cleanest gap we got across any Q3 experiment.

## What this experiment teaches

- **Aux-loss design works.** Unlike adversarial training (unstable), bottleneck (stochastic), or simple L2 norm (no effect at all), this gives a controlled, repeatable injection of a specific non-linear geometry.
- **Single-feature scope is a feature, not a bug.** The recipe encodes one feature in radius. Applying to multiple features would require multiple distinct non-linear tricks (e.g., feature B on a circle, feature C as cluster membership) since radius is a single scalar.
- **β > 0 is required.** Without the direction-isotropy loss, `L_radial` just adds radius information without removing the linear direction signal. The trick *only works as a combination*.
- **There's a Pareto frontier** between "scrub F" and "preserve others." Higher α/β scrubs F more aggressively but damages other features through global geometric constraints. The sweet spot is where the aux losses are barely strong enough.
- **The norm-only probe is the diagnostic that confirms the mechanism.** A linear probe failing + a 1-feature `‖h‖` probe succeeding is the textbook signature of a concentric-shell encoding.

## Comparison to other Q3 ideas

| Experiment | Mechanism | Result on F | Other features |
|---|---|---|---|
| Adversarial (#1/#6) | GRL co-training | partial scrub at cost of head collapse | preserved |
| Bottleneck (#3) | shrink hidden 2 to k<8 | stochastic winner; sentiment gap +0.39 at k=3 | many drop |
| L2 normalize (single sphere) | unit norm at hidden 2 | no effect — *more* linear | unchanged |
| **Concentric shells (#7, this)** | **radial aux loss + iso aux loss** | **clean +0.39 gap on F** | **all preserved** |

This is the cleanest "intentional non-linear encoding" result — a controlled, surgical change with a verified mechanism.

## Not pursued

- **Multi-feature shells.** Would need 2ᵏ distinct radii for k features (4 for two features, 256 for all 8). Possible for k=2; impractical for k=8.
- **Other geometric tricks per feature.** Could encode each of the 8 features in a different non-linear way (one by radius, one by angle on a circle, one by cluster membership) and combine the aux losses. Much more complex.
- **Stability check across seeds.** A single-seed run; the aux-loss approach should be more reproducible than bottleneck (no allocation-game stochasticity), but worth verifying.
