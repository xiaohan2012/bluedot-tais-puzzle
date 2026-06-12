# Q3 — Two moons (and other candidate shapes)

## Goal

Generalize the `q3_concentric_shells` recipe (`L_main + α·L_shape + β·L_iso`) to a target shape other than concentric shells. Two moons is the natural first choice: it's the most-familiar "non-linearly separable in 2D" benchmark, with a known sklearn implementation, and admits a clean 2D visualization.

If the recipe generalizes, swapping in any shape function `L_shape` should produce the same +0.4 probe/head gap on the target feature that concentric shells achieved. We started with moons and discovered it does **not** work out of the box.

## Attempts so far

### Attempt 1 — random target per batch (50 fixed crescent points)

Pulled `h2[:, :2]` to the **softmin-distance** to either crescent (50 fixed target points per crescent), randomly varying the chosen point per batch.

**Result (α=0.1, β=0.05, applying iso loss to dims 2:):**

| feature | probe | own-head | gap |
|---|---:|---:|---:|
| sentiment | 0.985 | 0.984 | −0.001 |
| (other 7 all 0.97+) | | | |

No effect. Probe ≈ head ≈ 0.99 for every feature.

### Attempt 2 — frozen per-sample target (sklearn `make_moons` style)

Per the snippet pattern: generate `N` moon coordinates via `make_moons(noise=0.05)`, assign each training sample a frozen random moon coord of matching class, pull `h2[:, :2]` to that exact target each batch.

**Result (α=0.1, β=0.05):**

| feature | probe | own-head | gap |
|---|---:|---:|---:|
| sentiment | 0.981 | 0.980 | −0.001 |
| (other 7 all 0.97+) | | | |

Still no effect. Cranking α to 1.0 and β to 0.5 collapsed `number` and `body_part` but **did not move sentiment's probe accuracy** (0.976).

### Diagnostic — where did sentiment survive?

| probe slice | sentiment test acc |
|---|---:|
| Linear probe on dims 0–1 only | 0.975 |
| Linear probe on dims 2: only | 0.618 |
| (placebo: random labels from dims 0–1) | 0.513 |

Statistics on test activations:

```
h2[:, 0]: mean ≈ 0, std = 0.74
h2[:, 1]: mean ≈ 0, std = 0.40
target moon: x ∈ [−1, 2],  y ∈ [−0.5, 1]
```

`L_shape` stayed at 0.6–0.9 throughout training — the model is **not** placing samples on the moon shape. Dims 0–1 are compressed to a small region where the two classes sit on opposite sides (linearly separable).

## Why per-sample frozen targets degrade to centroids

For sample `i` of class c, the frozen target is **one random moon coord** of class c. The model must map `sentence_embedding_i → moon_coord_i`. The moon coord is assigned at random; nothing in the input determines *which* point on the moon a sample maps to.

Under MSE, the loss-minimizing prediction is the **per-class conditional mean of the targets** — the centroid of each moon (roughly `(0, 0.5)` for upper, `(1, 0)` for lower). These centroids are linearly separable, so the result is a trivial 2-cluster encoding that linearly separates sentiment.

The snippet's approach (from sklearn's two-moons classification) works because *there* the input **is** the moon coordinate — the model trivially learns identity. In our setup the input is a sentence embedding, unrelated to the random target assignment, so the model averages to the centroid.

**Compare to concentric shells (which works):**

The per-sample target there is just a **radius** `target_R[y_F]`, not a point. That leaves 63 angular degrees of freedom completely unconstrained. The model can use direction for whatever other features need, while still hitting the radius. Shape constraint and multi-task objective don't fight.

## Open questions

- Would a **manifold-distance loss** (chamfer / nearest-point on the assigned moon curve) replace the unlearnable point-target with a learnable "stay on the curve" objective? The model could pick where on the moon to land per sample.
- Would a **learned 2D projection** (`Linear(64, 2)` on top of hidden 2, with the shape loss applied to the projection) help by letting the model choose which 2D subspace to use for the moon — instead of fixing it to dims 0–1?
- Is the failure mode specific to moons (mild non-linearity ceiling at ~85% linear separability), or would the same architecture fail with stronger non-linear shapes like spirals or checkerboards?

## Other candidate shapes (not yet implemented)

The recipe — `L_main + α·L_shape + β·L_iso` with `L_iso` scrubbing the linear direction — should generalize to any target geometry where a hyperplane fails to separate the two classes. Candidates ordered roughly by expected difficulty of linear separation:

- **Nested rings (multi-shell).** Generalization of concentric shells with radii alternating between two values across multiple concentric layers: `R₀, R₁, R₀, R₁, …`. Linear probe near chance for any number of rings (radius is still the only class signal). Quantitative knob: number of rings.
- **Sphere-cap clusters.** F=1 lives in K disjoint small caps on a sphere; F=0 fills the rest. Linear probe fails when K ≥ 2 (no half-plane covers disjoint caps cleanly). Closest cousin of the original puzzle's `country` quadric.
- **XOR / checkerboard on 2 dims.** Tile a 2D plane into N×N alternating squares; F=1 in the "even" cells, F=0 in the "odd" cells. Polynomial probes of degree < N fail; degree N succeeds.
- **N-fold rotational parity.** F is the parity of an angular wedge index (0..N−1) — `even` for F=0, `odd` for F=1. Polynomial degree needed scales with N.
- **Spirals.** Two interleaved Archimedean spirals, one per class. Even high-degree polynomials struggle as the number of spiral turns grows.
- **Möbius strip / non-orientable surface.** F flips as you traverse a closed loop on the strip. Genuinely exotic topology; hardest to verify because no single low-dim visualization makes non-orientability obvious.

Expected behaviour given the shells/moons contrast:

- **Maximally non-linear** shapes (rings, sphere-cap clusters, spirals) should reproduce the clean +0.4 gap from shells.
- **Mildly non-linear** shapes (moons, low-N checkerboard / parity) will probably underperform — the linear probe ceiling is too high to scrub completely.

## Attempt 3 — moment-based shape losses (the working approach)

After the per-sample-target experiments failed, we switched to a fundamentally different design: instead of pinning each sample to a specific point, **constrain the angular distribution of each class via its moments** (radius, mean direction, covariance). The architecture also changed:

- **Hidden 2 is configurable width `k`** (slider). Setting `k=2` makes the displayed 2D scatter equal to the full representation — no slice ambiguity.
- **`HeadMoons`** has `Linear(384, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, k)` (no ReLU at the bottleneck — signed), then `Linear(k, 32) → ReLU → Linear(32, n_predict)`.
- The head outputs *only* the features in `predict_features` (multi-select), so this is a focused, often single-task experiment.

### Loss components

Each is computed per-class with `dirs = (z − own_center) / ‖z − own_center‖`:

1. **`L_radial = ((‖z − own_center[y]‖ − R)²).mean()`** — pull each class to its own circle of radius R.
2. **`L_dir = ‖mean(dirs|class 0) − target_d_0‖² + ‖mean(dirs|class 1) − target_d_1‖²`** — pull each class's mean unit direction toward a class-specific target.
3. **`L_unif = ‖cov(dirs|class 0) − target_cov‖_F² + ‖cov(dirs|class 1) − target_cov‖_F²`** — pull each class's covariance toward what a uniform-on-hemisphere distribution would have.

Optional / abandoned terms:
- **`L_opp = ‖mean_d0 + mean_d1‖²`** (opposition) — became redundant once we set opposite `target_d_0 = −target_d_1`, since satisfying `L_dir` implies opposition. Removed.

Total: `L = L_main + α·L_radial + β·L_dir + δ·L_unif`.

### Geometry — change 1 (per-class direction targets)

We picked the **last axis** `e` of hidden 2 as the hemisphere axis. Targets:

```
target_d_0 = +c_k · e          # class 0 — upper hemisphere along e
target_d_1 = −c_k · e          # class 1 — lower hemisphere
c_k = Γ(k/2) / (√π · Γ((k+1)/2))      # expected |mean direction| for uniform on hemisphere
```

`c_k = 0.637 (k=2), 0.5 (k=3), 0.424 (k=4), 0.249 (k=16)`. The non-zero target magnitude `c_k` is what *uniform-on-hemisphere* would produce — so a single-point clump (`|mean|=1`) is penalized for overshooting, and a full-circle (`|mean|=0`) is penalized for undershooting.

### Geometry — change 2 (interleaving offset)

```
center_0 = (0, 0, ..., 0)
center_1 = (1, 0, ..., 0, 0.5·R)
```

The `0.5·R` shift along the hemisphere axis is what creates the **sklearn-style interleaving**: class 0's upper hemisphere (`y ∈ [0, R]`) and class 1's lower hemisphere (`y ∈ [−0.5·R, 0.5·R]`) overlap in `y ∈ [0, 0.5·R]`. Without this shift, a horizontal hyperplane at `y=0` separates them perfectly.

### Uniformity — covariance-toward-isotropic (third moment)

For uniform on the upper hemisphere of S^(k−1) along `e`:

```
target_cov = (1/k) · I − target_d ⊗ target_d
           = diag(1/k, 1/k, ..., 1/k, 1/k − c_k²)
```

Off-diagonals are zero by reflection/rotational symmetry. For k=2: `diag(0.5, 0.095)` — wide horizontal spread, narrow vertical. For k=3: `diag(1/3, 1/3, 0.083)` — wide in the equatorial plane, narrow along e.

### Results

All at default `α=1.0`, `R=1.0` unless noted:

| state | k | β | δ | probe | head | gap | shape |
|---|---:|---:|---:|---:|---:|---:|---|
| Changes 1 + 2 only | 2 | 2.0 | 0 | 0.978 | 0.981 | +0.003 | two facing arcs, **clumped near hemisphere peaks** |
| + covariance (δ=5) | 2 | 2.0 | 5 | 0.980 | 0.977 | −0.003 | shape spreads more but probe doesn't drop |
| + covariance (δ=50) | 2 | 2.0 | 50 | 0.961 | 0.958 | −0.003 | shape closer to target but task loss collapses in lockstep |
| + covariance (δ=200) | 2 | 2.0 | 200 | 0.793 | 0.799 | +0.007 | **collapse** — both classes spread over same band, classes mix |
| Lowering β (no δ) | 8 | 0.5 | 0 | 0.954 | 0.972 | **+0.018** | tips fill in, real probe/head gap |

### Why moments aren't sufficient

The c_k target is a **first-moment** constraint. Two antipodal sub-clumps (or any bimodal distribution that averages to `c_k · e`) satisfy it without being uniform — and the model finds those because they're cheaper for the task loss. Diagnostic at the β=2 state:

```
class 0: |mean_d| = 0.661 ≈ c_k ✓ but variance is concentrated near a single peak
class 1: |mean_d| = 0.706 ≈ c_k ✓ same problem
```

Adding the second-moment (covariance) constraint helped a bit (`δ=50` pushed cov_yy from 0.063 → 0.077 toward target 0.095) but never opened a probe/head gap — the cov constraint and the task loss are in **direct tension**: cov wants samples spread uniformly, task wants samples placed where they're easy to classify.

At very high `δ` the cov term wins but destroys task performance. At moderate `δ` neither wins.

### Lowering β was the unlock

Counter-intuitively, **reducing the mean-direction weight from β=2 to β=0.5 was the most effective single change**. With β=2 the model strongly fits the c_k target which pulls each class to the centroid of its hemisphere (the "peak" of the half-arc). Lowering β to 0.5 relaxes that pull — samples flow toward the *tips* of the half-arc, which is where the genuine interleaving happens.

At k=8, β=0.5: probe drops to 0.954 with head at 0.972 — **the first non-trivial probe/head gap (+0.018) in the moons experiment**.

### Geometry tweaks that don't help

We also tested two geometric knobs as alternatives:
- **R = 1.5 (bigger arcs):** probe 0.973, head 0.977 — no improvement. Uniform scaling preserves the relative arrangement.
- **x-offset = 0.6 (tighter interleaving):** probe 0.981, head 0.983 — no improvement. Moving from sklearn's `1.0` ratio toward stronger overlap didn't open a gap.

The geometry of changes 1 + 2 already matches sklearn moons. The issue was always **how mass distributes within each hemisphere**, not the hemispheres themselves. Lowering β addresses that directly.

### What we learned

- **Moment-based losses are a real alternative to per-sample targets.** They avoid the random-assignment degeneracy because they constrain *distributional* properties, not specific point assignments.
- **First moments alone are insufficient.** Two antipodal clumps satisfy `mean = c_k · e` exactly, and the task loss happily exploits this.
- **Second moments help a little but trade off with the task loss.** Cov-toward-isotropic moves the distribution toward the desired shape but the task loss resists.
- **The mean-direction weight is the key tunable.** Strong `β` forces samples to the hemisphere peaks (linearly separable); weak `β` lets samples reach the tips (interleaving fills in).
- **Higher `k` makes the 2D scatter misleading.** At k=3 each class projects to a *filled half-disk* instead of a thin arc; at k=8 we see a 2D slice of an 8D shape. The probe operates on the full k-dim representation, so the probe number is honest, but the visual diagnostic loses fidelity. For shape inspection, k=2 is the only fully faithful setting.

### Open question — true uniformity

Even with `β=0.5`, the +0.018 gap is far short of the +0.4 gap concentric shells achieved. The remaining gap is the same fundamental issue: a hyperplane in k-dim can still separate two half-shells with the right mass distribution, even after lowering β. The genuinely-hard moons require samples spread *uniformly* over the assigned half-arc — including significant mass at both tips simultaneously — which neither first- nor second-moment constraints can pin down exactly.

Possible next experiments:
- **Wasserstein distance to a discrete uniform reference set** (N points evenly spaced on the target half-arc) — strongest distributional constraint, expensive.
- **Tip-anchor loss** — explicitly pull a subset of samples to predefined tip locations. Ad hoc but direct.
- **Combine `β=0.5` with a small covariance term (δ ∈ [10, 50])** — the two relaxations might be complementary; one frees the bulk to spread, the other shapes the variance.
- **Accept the partial result and write up.** The recipe shows the principle (geometric injection of non-linear structure for a chosen feature), even if the linear-separability ceiling for moons is intrinsically higher than for shells.

## What to try next

Per the open questions above, the most promising next experiment is the **branched architecture with a learned 2D projection**:

```
hidden_2 (64D)
   ├── main path: → Linear → ReLU → Linear → logits          (BCE main loss)
   └── shape branch: → Linear(64, 2) → projection            (moon shape loss)
```

Combined with a **chamfer-style "stay on the moon curve" loss** instead of per-sample point targets, so the random-assignment problem disappears. The model chooses (a) which 2D subspace to put the moon in, (b) where on the moon to land each sample.

If this still fails, the conclusion is that moons (mild non-linearity) is a poor target for this recipe regardless of architecture, and we should switch to a maximally-non-linear shape (rings or sphere-cap clusters) to confirm the recipe generalizes.
