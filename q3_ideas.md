# Q3 — Ideas for a "weirder" representation

The puzzle's Q3 is open-ended: "more interesting" is ours to define and defend. The original model's quadric encoding of `country` is interesting because the boundary is a smooth curved surface — a clean departure from linear directions. To beat it, we should pick a dimension of "interestingness" and push hard on it.

Below: candidate dimensions, each with the core idea, why it's interesting, and a rough training recipe.

---

## 1. Harder separability (high-degree polynomial)

- **Idea:** make F require a polynomial of degree ≥ k to separate (quadric was k=2).
- **Why interesting:** a clean, quantitative weirdness metric — show that degree-(k−1) probes fail while degree-k succeeds.
- **Recipe — XOR composition:**
  - Relabel: F = `f₁ XOR f₂ XOR … XOR fₖ` for k ≥ 3 of the existing features (e.g. `country XOR person XOR question`).
  - Retrain the same 5-layer MLP head with this new F replacing one of the 8 targets.
  - XOR forces degree-k separability — model can't shortcut it.
- **Test:** show degree-(k−1) poly LogReg fails, degree-k succeeds, MLP succeeds.

## 2. Topologically structured (non-trivial manifold)

- **Idea:** encode F on a circle, torus, sphere, or Möbius strip.
- **Why interesting:** mirrors Engels et al. "Not All Language Model Features Are Linear" (days-of-week on a circle). Genuinely novel geometry, not just "harder polynomial."
- **Recipe — geography circle:**
  - Replace `country` binary label with a continuous angle θ derived from each country's longitude (or any cyclic property).
  - Train with a regression head that decodes (sin θ, cos θ). Model is forced to encode country on a circle.
  - For a binary version: bucket angle into 2 halves but reward the model only if its hidden 2 activations form a coherent ring.
- **Test:** PCA top-2 of hidden 2 activations for country-bearing texts should show a circular layout. Probe with `arctan2(PC1, PC2)` ≈ θ.

## 3. Superposed / distributed encoding

- **Idea:** encode F across many dimensions such that no low-rank projection reveals it. Sparse-coding-style superposition (Anthropic toy models).
- **Why interesting:** matches the empirical finding that real LLMs encode many more features than they have neurons by overlapping directions.
- **Recipe — bottleneck + sparsity:**
  - Shrink hidden 2 from 64 to e.g. 16 units while keeping the 8-feature output target.
  - Add an L1 penalty on hidden 2 activations to encourage sparsity.
  - The model must pack 8 features into 16 dimensions, forcing overlap; F lives in a multi-direction code.
- **Test:** show that no rank-1, rank-2 linear probe recovers F, but a sparse autoencoder on hidden 2 finds the F direction.

## 4. Compositional / hierarchical

- **Idea:** F is computed by a multi-step routine — e.g., model first builds a latent that combines two sub-features, then reads F from the latent.
- **Why interesting:** small mechanistic circuit, more "model is reasoning" than "model is geometrying."
- **Recipe — latent intermediate:**
  - Make F depend on two latent properties (e.g., F = (vowel_count > 3) AND (sentence_starts_with_capital)).
  - These intermediates aren't given as separate features; the model must construct them.
  - Optional: add a probe-suppression loss that pushes hidden 2 to *not* directly encode F linearly, only via the latents in hidden 1.
- **Test:** show F lives in hidden 3 but not hidden 2; ablating specific hidden-1 directions destroys F.

## 5. Timing / sparsity-based (which neurons, not how much)

- **Idea:** encode F in *which subset* of neurons fire, not in continuous magnitudes — closer to discrete codes.
- **Why interesting:** departs from the continuous-direction paradigm entirely; aligns with sparse autoencoder dictionary atoms.
- **Recipe — top-k activation:**
  - Add a top-k activation function at hidden 2 (only the top-k of 64 neurons survive, rest zeroed).
  - With strict k, F must be encoded as a specific *combination* of which neurons fire.
- **Test:** F is recoverable from the binary mask of which neurons are active, but the magnitudes carry little info.

## 6. Adversarially probe-resistant

- **Idea:** explicitly defeat linear, polynomial-deg-2, and shallow-MLP probes; only a specific architecture recovers it.
- **Why interesting:** stress-tests the whole probing toolkit; useful as a benchmark for interp tools.
- **Recipe — adversarial co-training:**
  - At each step, train probes (linear, poly2, MLP) on hidden 2; add a loss that *maximizes* their error while still letting the model's own head solve F.
  - Iterate until all standard probes fail but the model's full readout succeeds.
- **Test:** demonstrate large gap between probe acc and full-model acc on F.

---

## Recommended exploration order

1. Start with **#1 (XOR composition)** — fastest to implement, cleanest result, directly parallels our Q2 quadric analysis.
2. Then **#2 (topological / circle)** — most aesthetically interesting, closest to published Engels-style findings.
3. **#3 (superposition)** if there's time — different flavor of "weird," addresses a different research question.
4. #4–#6 are more involved and best as stretch goals.
