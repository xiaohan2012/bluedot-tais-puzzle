# Report

## Introduction

This report documents our exploration of the three puzzle tasks. Code and notebooks are available at [github.com/xiaohan2012/bluedot-tais-puzzle](https://github.com/xiaohan2012/bluedot-tais-puzzle).

It is organized as follows.

- **`country` feature cannot be linearly separated** (Q1): we identify `country` as the feature that is not linearly represented at hidden layer 2, using per-feature linear probes.
- **How is country feature represented?** (Q2): we show that `country` is separable by a quadratic surface, supported by a 2D PCA visualization and a degree-2 polynomial probe.
- **Other interesting representations** (Q3): we explore three ways to induce non-linear representations at hidden layer 2 — capacity pressure via a bottleneck, concentric shells via auxiliary losses, and an adversarial gradient-reversal attempt (which fails, with reasons named).

## `country` feature cannot be linearly separated

`country` is the feature that is not represented linearly at that specified layer activations.

The detection method is straight-forward:

1. For each feature, fit a linear probe (we chose logistic regression) on the training data
2. and report the accuracy on the test set

For the given dataset, we get the following test scores by the linear probes:

*Table 1: Per-feature test accuracy of linear probes on hidden-layer-2 activations. `country` stands out as nearly chance-level.*

| Feature     | Test accuracy |
| ----------- | ------------: |
| number      |         0.975 |
| question    |         1.000 |
| color       |         0.971 |
| food        |         0.985 |
| sentiment   |         0.982 |
| **country** |     **0.427** |
| person      |         0.998 |
| body_part   |         0.980 |

Linear probes perform poorly on `country` -- its success relies largely on chance, whereas for other features, test accuracy is close to 1.0.

We complement the above result with the per-class distribution of predicted log-odds. For the sake of space, we pick just one feature (`person`) from the linearly separable feature group.

![[images/Pasted image 20260610221314.png]]

*Figure 1: Per-class log-odds distributions from the linear probe — `country` (left) is fully mixed, `person` (right) is cleanly bi-modal.*

We can see that:
- linearly separable features (`person`, on the right) shows well-separated bi-modal log-odd distributions
- whereas for `country` feature, linear probe offers no such distinction -- predictions on the two classes are completely mixed up.

We will see in the next section that using more complex probes (e.g, quadratic probe) can pull the distributions apart for `country`.


## How is country feature represented?

We find that the `country` feature can be separated by quadratic surfaces (for example, an ellipse).

To get an intuition, we project the data points into 2-dimensional space using PCA (where top-2 principal components explain ~78% of the total variance). As shown below, the sandwich shape suggests that a quadratic surface can separate the points nicely.

![[images/Pasted image 20260610215300.png]]

*Figure 2: 2D PCA projection of hidden-layer-2 activations colored by `country`. The sandwich shape hints at quadratic separability.*

We further validate the quadratic separability by replacing the linear probe with a quadratic one. 

The only difference between the two probes is that quadratic probe introduces 2nd-degree feature interactions. In `sklearn`, it is implemented as

```python
make_pipeline(
	PolynomialFeatures(degree=2, include_bias=False),  # without PolynomialFeatures, it reduces to linear probe
    LogisticRegression(),
)
```

Using this probe, test accuracy on `country` jumps to **0.931**, much higher than before (0.427).

The performance boost is also backed by the bi-modal shape in its log-odds distribution. We show a comparison plot between linear probe and degree-2 polynomial probe on `country` feature below

![[images/Pasted image 20260610223740.png]]

*Figure 3: Log-odds distributions on `country` — linear probe (left) stays mixed, while the degree-2 polynomial probe (right) separates the two classes.*

## Other interesting representations

Each subsection below describes an exploration on a specific type of representation.

- **Capacity pressure inspired by superposition**: shrink hidden layer 2 to force features to share neurons.
- **Concentric shells**: push the two classes onto nested shells around a shared centroid via auxiliary losses.
- **Towards general linear inseparability via gradient reversal (a failed attempt)**: adversarially discourage *any* linearly separable representation using a gradient-reversal layer.
### Capacity pressure inspired by superposition

In this experiment, we reduce the size of the 2nd hidden layer, which we studied in the previous tasks, in order to introduce "capacity pressure".

This design is motivated by observations from the previous model.

**Observation 1**: **activations are highly sparse** -- only 11 out of the 64 neurons fire, while the remaining are effectively dead. Below shows the mean and standard deviation of activations over the training data.

*Table 2: Mean and standard deviation of each hidden-layer-2 neuron's activation. Only 11 of 64 neurons fire; the rest are dead.*

| neuron index  | mean     | std      |
| ------------- | -------- | -------- |
| 34            | 2.53     | 0.64     |
| 46            | 1.74     | 0.63     |
| 10            | 1.37     | 0.62     |
| 36            | 2.10     | 0.53     |
| 43            | 1.23     | 0.38     |
| 55            | 1.46     | 0.36     |
| 19            | 1.21     | 0.35     |
| 51            | 1.10     | 0.31     |
| 1             | 0.86     | 0.24     |
| 44            | 0.31     | 0.17     |
| 23            | 0.42     | 0.16     |
| **remaining** | **0.00** | **0.00** |

**Observation 2:**  **activations show superposition**, indicated by the overlapping linear probe coefficients.

Below shows the coefficient heat map of the linear probes on all features other than `country`. A coefficient quantifies the importance the associated neuron plays in predicting a specific feature. 



![[images/Pasted image 20260611092510.png]]

*Figure 4: Heat map of linear-probe coefficients across alive neurons and features. Overlapping patterns indicate polysemantic neurons and distributed feature codes.*

We can see that one feature may fire multiple neurons, and one neuron is associated with multiple features. For example, 

- `sentiment` is decoded from at least 5 neurons — 1, 19, 36, 46, and 55.
- Neuron 19 carries non-trivial weight for four different features — `question`, `food`, `sentiment`, and `body_part`.

The above observations make us wonder: what if we reduce the size of that hidden layer, to explicitly enforce superposition, what will happen?

In the experiment below, we vary the size of that hidden layer (denoted as `k`) in the range of `[2, 16]` and count each of the following cases:

- **both** the linear probe accuracy $\ge \tau$  and the head accuracy $\ge \tau$ -- representations are already linearly separable at that hidden layer
- **head only**: head accuracy  $\ge \tau$  but linear probe  $< \tau$  -- representations are non-linearly separable at that layer, but separable in the end
- **neither**: both models' accuracy $< \tau$  -- they fail to learn this feature at this `k`

We set $\tau=0.85$.

![[images/Pasted image 20260611201302.png]]

*Figure 5: Per-feature outcome across `k`. As `k` shrinks, the share of "head only" and "neither" cases grows, showing capacity-driven non-linear encoding.*

We can see that the value of `k` directly imposes a capacity bottleneck on the whole model's capability, not just the hidden layer being studied. For example: 
- When `k` is large enough (e.g, `k >= 13`), both the head and the hidden layer learn clearly separable representations (the long green bar) across different features
- However, as `k` decreases, more features tend to suffer from inseparable learned representations (the elongating red bars).

Perhaps what is more interesting is the occasional appearance of head-only case, that is, the model fails to learn linearly separable representation at hidden layer 2, but the final predictions by model head are accurate. This scenario mirrors what we see in the previous two tasks. 

For example when `k=5`, the embeddings of `question` and `food` in 2D space are visualized below. `question` is linearly separable by the probe, but `food` is not

![[images/Pasted image 20260611202432.png]]

*Figure 6: 2D embeddings at `k=5` for `question` (linearly separable) and `food` (not separable by a linear probe, yet correctly classified by the head).*

### Concentric shells

Next, we explicitly control the shape of the hidden representation by modifying the loss function. Specifically, we add two loss terms that explicitly encourage the representations to form concentric shells, illustrated below:

![[images/Pasted image 20260611214655.png]]

*Figure 7: Target geometry — two classes occupy concentric shells around a shared centroid at distinct radii.*

To form the concentric structure, there are three ingredients:
1. Class centroids (mean of activations within a class) are close to each other, so they're concentric
2. Points of different classes are spread away with different distance to their centroids
3. Points of the same class are equally distant to their centroid

Next we describe how to encode the above intuition into the training loss function. Denote two distance/radius values `R1` and `R2`, which specifies how far points of each class are from the origin. 

**Centroid gap loss** captures the 1st ingredient. It measures the distance between the two classes' centroids. That is,

`L_centroid_gap = ‖centroid(F=1) − centroid(F=0)‖²` 

**Radial loss** captures the 2nd and 3rd ingredient. It is the mean squared error between hidden activations' distance to the class centroid and the expected radius, that is, 

`L_radial = mean((‖h2_c‖ − target_R)²)`, where

-  `h2_c` is the batched-centered activations on hidden layer 2
-  `target_R` is target radius value (`R1` or `R2` depending on the sample's feature value)

In other words, `L_radial` pulls `y=1` samples to radius `R1` and `y=0` samples to `R2`.

**Training loss** is simply the weighted sum of the 3 terms, where 
- `alpha` and `beta` control the relative strength of the two new terms
- `L_main` is the task-specific binary cross entropy loss

```
L = L_main + alpha x L_radial + beta x L_centroid_gap
```

**Injecting linear inseparability:** using this technique, we are also able to produce a remarkable performance gap between linear probe and the final head.

For example, when imposing the radial loss and centroid gap loss on `sentiment` (selected for illustration), and setting `alpha=0.1` and `beta=0.05`, the linear probe's test accuracy on `sentiment` is 0.59, whereas the head's test accuracy is 0.98.

![[images/Pasted image 20260611215645.png]]

*Figure 8: PCA of hidden-layer-2 activations under concentric-shells training on `sentiment` (`alpha=0.1`, `beta=0.05`). The two classes form nested rings — linearly inseparable but quadratically separable.*

**Ablation study**: last we show both radial loss and centroid gap loss are indispensable to achieve the desired outcome of linear inseparability. We re-use `sentiment` as the intervened feature.

Removing `L_radial` (by setting `alpha=0`) gives no performance gap between linear probe and model head. Their accuracy scores are both 0.981. The intuition is that the points are no longer pulled towards different distances, which is essential for linear inseparability

Below shows the resulting PCA visualization -- no radial distance penalty anymore, therefore points tend to be scattered everywhere.

![[images/Pasted image 20260611223849.png]]

*Figure 9: PCA after ablating `L_radial` (`alpha=0`). Without the radial pull, the two classes are linearly separable along the centroid-gap axis.*

Removing `L_centroid_gap` (by setting `beta=0`) also gives no performance gap -- both models' accuracy scores are close to 1.0. Meanwhile an interesting observation is that in the 2D PCA plane the shells look perfectly concentric. 

In 64-D they aren't concentric: the inner and outer clouds are also separated along a hidden direction, which the PCA projection squashes onto the same plane.

![[images/Pasted image 20260611224016.png]]

*Figure 10: PCA after ablating `L_centroid_gap` (`beta=0`). Rings look concentric in 2D, but the classes remain linearly separable along a hidden direction the PCA projection collapses.*

TODO: might be better to create a 3D plot

### Towards general linear inseparability via gradient reversal (a failed attempt)

Though we fail to produce the desired outcome, we still describe what we have tried.

We borrow the idea from *gradient reversal*, a concept in adversarial machine learning, and attempt to discourage linear probes from performing well during the training process. 

In contrast to concentric shells, this approach does not tell what specific geometry the hidden representation should form, instead, it just says "do not form linearly separable representations". Therefore, the resulting geometry is potentially "general".

In this approach, we train the model to deliberately learn hidden representations that cause linear probes to fail. Specifically,

- A proxy linear probe is used during the training process. At each training iteration, the linear probe's model weights are updated in an attempt to perform well on the classification task
- However, instead of letting the probe's gradients improve the representation, we *reverse* the sign of the gradient, so that the representation makes the probe's task harder.

**How gradient reversal works.** Technically, the loss function has two parts: 1) the normal task loss, and 2) the loss on the probe's performance.

```
L_main  = BCE( main_head(h),    y_all )         # main task loss
L_probe = BCE( linear_probe(h), y_F   )         # probe's loss on feature F
```

The linear probe is trained to minimize `L_probe`. The encoder + main head minimize the combined loss below, where the probe's gradient is sign-flipped before it reaches the encoder:

```
L = L_main + lambda x GRL( L_probe )
```

where

- `h` is the hidden representation at hidden 2.
- `y_F` is the binary label of the target feature `F`.
- `y_all` is the full 8-feature label vector.
- `lambda` controls the strength of the adversarial pressure.
- `GRL` is the *gradient reversal layer*: identity in the forward pass, sign-flip in the backward pass.

We expect linearly non-separable representations to form, and therefore worse performance of linear probes than the head. However, we didn't achieve our goal. The following observations are made.

**Observation 1**: the proxy/in-loop linear probe's accuracy is ~0.5, but the held-out probe's (the logistic regression model trained on hidden representations) accuracy is much higher. And the big gap holds throughout the whole training process, as shown below.

This may suggest that the in-loop linear probe is not a faithful reflection of the held-out one.

![[images/Pasted image 20260612194904.png]]

*Figure 11: In-loop probe accuracy stays near chance while a fresh held-out probe recovers `sentiment` throughout training — evidence the encoder is dodging rather than scrubbing F.*

**Observation 2**: head performance is *worse* than the linear probe, which is contrary to what we expect. This is perhaps the biggest surprise to us.

![[images/Pasted image 20260612201929.png]]

*Figure 12: Per-feature test accuracy after GRL training on `sentiment`. The head (hatched) is worse than a held-out linear probe (solid) on the target feature — the opposite of the intended gap.*

## Acknowledgement on AI usage

- Claude Code (Opus 4.7) is used for code writing, experimentation, and proof-reading the final report.
- Claude (Opus 4.7) and Gemini (Pro) are used for brain-storming.