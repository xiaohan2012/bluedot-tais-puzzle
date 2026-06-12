# Findings

## Q1 — Which feature is non-linear at hidden 2?

**`country`.**

Linear probe (LogReg, fit on 7000 train, eval on 1500 test) at post-ReLU hidden 2:

| feature     |  test acc |     |
| ----------- | --------: | --- |
| number      |     0.975 |     |
| question    |     1.000 |     |
| color       |     0.971 |     |
| food        |     0.985 |     |
| sentiment   |     0.982 |     |
| **country** | **0.427** |     |
| person      |     0.998 |     |
| body_part   |     0.980 |     |

- All features except `country` are linearly separable (≥ 0.97).
- `country` sits at chance — a single direction cannot recover it.
- At hidden 3, all 8 features become linearly separable (the final `Linear(64→8)` reads them all off linearly by construction), confirming hidden 2 is where the non-linear encoding lives.

## Q2 — How is `country` represented at hidden 2?

**Approximately a quadric surface in 64-dim activation space.**

Probe comparison on `country` (train/test):

| probe                                              | test acc |
|----------------------------------------------------|---------:|
| Linear LogReg                                      |     0.43 |
| Distance to country=1 centroid (1 feature, LogReg) |     0.73 |
| **Degree-2 polynomial LogReg**                     | **0.93** |
| Shallow MLP (1×32)                                 |     0.96 |

- **Quadric boundary:** degree-2 polynomial features lift accuracy from 0.43 to 0.93 — the decision surface is well-approximated by a single quadratic surface.
- **Not a pure sphere:** distance to the country=1 centroid alone reaches only 0.73, ruling out a simple radial encoding around one point. The structure is anisotropic — likely an ellipsoid / paraboloid.
- **Visual signature on t-SNE:** country=1 forms an interior blob surrounded by country=0 — consistent with an ellipsoidal envelope where country=1 lives inside and country=0 lives outside.

## Sanity check: how are the 7 *linear* features encoded?

Linear-probe coefficient heatmap (7 features × 64 hidden-2 neurons, each row max-normalized):

- **Sparse readout.** Each row has 1–3 bright cells; top-5 neurons capture **65–84%** of the L1 mass per feature.
- **Privileged basis.** Bright cells consistently land on a small fixed set of column indices — roughly **{1, 10, 19, 23, 34, 36, 44, 46, 51, 55}** — about 10 of 64 neurons do most of the linear work. If the basis were arbitrary, no specific column would stand out across rows. The post-ReLU at hidden 2 makes the basis privileged (rotation-non-invariant), and the model exploits it.
- **Polysemantic neurons.** Several neurons appear bright across multiple features (e.g. neuron 19 → question, sentiment, body_part; neuron 55 → question, sentiment, person). Consistent with superposition: more directions than neurons, packed by overlap.
- **~50/64 neurons are dark for the 7 linear features.** Capacity that is plausibly used to carve out `country`'s quadric surface.

Caveats:
- Probe coefficients show what *decodes* a feature, not necessarily everything that *encodes* it (a neuron with `|w|≈0` could still co-fire redundantly with another). A per-neuron AUC analysis would corroborate.
- Row normalization is for visual comparability across features; raw scales differ.

## Method notes

- Activations: raw post-ReLU at hidden layer 2 (slice `m.layers[:6]`).
- Probes fit on `data/train.jsonl` (7000), evaluated on `data/test.jsonl` (1500).
- MLP regularized at `alpha=1e-2` to discourage memorization.
- Class balance: every feature is ~50/50 in train — class weighting unnecessary.
