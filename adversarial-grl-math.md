# Adversarial co-training with GRL — the loss function

A reference note on the mathematical form of the adversarial probe-resistance objective and its equivalence to the first-moment-matching aux loss (`L_iso`).

## Notation

- $x$ — input embedding; $h = E_\phi(x)$ — encoder output (hidden 2). $\phi$ = encoder parameters.
- $\hat y_\text{all} = H_\psi(h)$ — main 8-feature head with parameters $\psi$, target $y_\text{all}$.
- $\hat y_F = P_\theta(h) = w^\top h + b$ — linear probe with parameters $\theta = (w, b)$, target $y_F$.
- $\ell_\text{BCE}$ — binary cross-entropy.

## Two component losses

$$
\mathcal L_\text{main}(\phi,\psi) \;=\; \mathbb E_{(x,y_\text{all})}\!\left[\ell_\text{BCE}\bigl(H_\psi(E_\phi(x)),\, y_\text{all}\bigr)\right]
$$

$$
\mathcal L_\text{probe}(\phi,\theta) \;=\; \mathbb E_{(x,y_F)}\!\left[\ell_\text{BCE}\bigl(P_\theta(E_\phi(x)),\, y_F\bigr)\right]
$$

## The minimax objective

$$
\boxed{\;
\min_{\phi,\psi}\;\max_{\theta}\;\;
\mathcal L_\text{main}(\phi,\psi)\;-\;\lambda\,\mathcal L_\text{probe}(\phi,\theta)
\;}
$$

with $\lambda > 0$ the GRL weight.

Read literally:

- Encoder + main head $(\phi,\psi)$ try to **minimize** the main loss **and increase** the probe loss (the $-\lambda$ in front of $\mathcal L_\text{probe}$).
- Probe $\theta$ tries to **minimize** its own loss — equivalently **maximize** the bracketed expression (since its own loss enters with a minus).
- At equilibrium $\theta$ is the **best** linear probe of $h$, yet it still can't beat chance — so $\mathbb E[h\mid F{=}1] \approx \mathbb E[h\mid F{=}0]$.

## GRL turns the minimax into a single SGD-style descent

Define a single scalar loss

$$
\mathcal L_\text{total}(\phi,\psi,\theta) \;=\; \mathcal L_\text{main}(\phi,\psi)\;+\;\lambda\,\mathcal L_\text{probe}\bigl(\text{GRL}(\phi,\lambda),\,\theta\bigr)
$$

where $\text{GRL}$ is identity in the forward pass and $-\nabla$ in the backward pass. All three parameter sets descend the *same* scalar $\mathcal L_\text{total}$, but the gradients route as:

$$
\nabla_\psi \mathcal L_\text{total} \;=\; \nabla_\psi \mathcal L_\text{main}
$$

$$
\nabla_\theta \mathcal L_\text{total} \;=\; \lambda\,\nabla_\theta \mathcal L_\text{probe}
\qquad(\text{probe minimizes its own loss})
$$

$$
\nabla_\phi \mathcal L_\text{total} \;=\; \nabla_\phi \mathcal L_\text{main}\;-\;\lambda\,\nabla_\phi \mathcal L_\text{probe}
\qquad(\text{encoder moves against the probe})
$$

This is why the implementation looks like a single `loss.backward()` call: GRL bakes the sign flip into the autograd graph so the minimax becomes a vanilla optimization.

## Connection to the aux-loss form (`L_iso`)

If $h$ is roughly Gaussian within each class, the optimal linear probe achieves a loss that depends on $\|\mu_1 - \mu_0\|$. Up to constants,

$$
\max_\theta \bigl(-\mathcal L_\text{probe}\bigr) \;\approx\;\; c\cdot \|\mu_1 - \mu_0\|^2
$$

so the GRL objective is asymptotically equivalent to

$$
\min_{\phi,\psi}\;\mathcal L_\text{main}(\phi,\psi) \;+\; \lambda c\,\|\mu_1 - \mu_0\|^2
$$

which is **exactly the `L_iso` form** used in the concentric-shells experiment. GRL on a linear probe and first-moment matching are the *same* objective in expectation — GRL just estimates it via a learned probe instead of a closed-form mean difference.

That equivalence is why we treat them as alternative implementations of the same idea, and why the simpler aux-loss recipe is preferred in practice: it converges in one step what GRL approaches as a noisy fixed point.
