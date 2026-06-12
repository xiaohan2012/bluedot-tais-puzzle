import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import json
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fnn
    from sentence_transformers import SentenceTransformer

    return Fnn, SentenceTransformer, json, mo, nn, np, torch


@app.cell
def _(mo):
    mo.md("""
    # Q3 — Two moons (2-D bottleneck)

    Force hidden 2 to be **literally 2-D** and pull it toward a two-moons shape
    via a frozen per-sample target (sklearn `make_moons` style). The
    downstream head is a small MLP `Linear(2, 32) → ReLU → Linear(32, out)`
    that can carve a curved decision boundary in the 2-D representation.

    No iso loss — the bottleneck is the only representational degree of
    freedom, so there's nowhere for the feature to hide.

    Diagnostics:
    - 2-D scatter of hidden 2 colored by the moon-feature label (does the moon shape actually emerge?)
    - linear probe vs model's own head accuracy per predicted feature.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Configuration
    """)
    return


@app.cell
def _(mo):
    feature_names_list = ["number", "question", "color", "food", "sentiment", "country", "person", "body_part"]

    moon_feature = mo.ui.dropdown(
        options=feature_names_list,
        value="sentiment",
        label="Moon feature (drives the per-sample shape target)",
    )
    moon_feature
    return feature_names_list, moon_feature


@app.cell
def _(feature_names_list, mo):
    predict_features = mo.ui.multiselect(
        options=feature_names_list,
        value=["sentiment"],
        label="Features the head predicts",
    )
    predict_features
    return (predict_features,)


@app.cell
def _(mo):
    alpha_slider = mo.ui.slider(
        start=0.0, stop=5.0, step=0.1, value=1.0,
        label="α — radial loss weight",
        show_value=True,
    )
    alpha_slider

    return (alpha_slider,)


@app.cell(hide_code=True)
def _(mo):
    beta_slider = mo.ui.slider(
        start=0.0, stop=5.0, step=0.1, value=1.0,
        label="β — direction-uniformity loss weight",
        show_value=True,
    )
    beta_slider

    return (beta_slider,)


@app.cell(hide_code=True)
def _(mo):
    k_slider = mo.ui.slider(
        start=2, stop=64, step=1, value=16,
        label="hidden-2 width k",
        show_value=True,
    )
    k_slider

    return (k_slider,)


@app.cell(hide_code=True)
def _(mo):
    R_slider = mo.ui.slider(
        start=0.2, stop=3.0, step=0.1, value=1.0,
        label="R — radius of each class's circle",
        show_value=True,
    )
    R_slider
    return (R_slider,)


@app.cell
def _(mo):
    mo.md("""
    ## Architecture
    """)
    return


@app.cell
def _(Fnn, nn):
    class HeadMoons(nn.Module):
        def __init__(self, n_out, hidden2_dim=2):
            super().__init__()
            self.h2_dim = hidden2_dim
            self.l1 = nn.Linear(384, 64)
            self.l2 = nn.Linear(64, 64)
            self.l3 = nn.Linear(64, hidden2_dim)          # hidden 2 is k-D, no ReLU
            self.head = nn.Sequential(
                nn.Linear(hidden2_dim, 32), nn.ReLU(),
                nn.Linear(32, n_out),
            )

        def hidden2(self, x):
            h = Fnn.relu(self.l1(x))
            h = Fnn.relu(self.l2(h))
            return self.l3(h)                             # (B, k), signed

        def forward(self, x):
            z = self.hidden2(x)
            return self.head(z), z


    return (HeadMoons,)


@app.cell
def _(mo):
    mo.md("""
    ## Data & embeddings
    """)
    return


@app.cell
def _(SentenceTransformer, json, np, torch):
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    with open("data/train.jsonl") as _f:
        _tr_rows = [json.loads(line) for line in _f]
    with open("data/test.jsonl") as _f:
        _te_rows = [json.loads(line) for line in _f]

    feature_names = json.load(open("feature_names.json"))
    tr_texts = [r["text"] for r in _tr_rows]
    te_texts = [r["text"] for r in _te_rows]
    tr_labels = np.array([r["labels"] for r in _tr_rows], dtype=np.float32)
    te_labels = np.array([r["labels"] for r in _te_rows], dtype=np.float32)

    tr_emb = torch.from_numpy(enc.encode(tr_texts, convert_to_numpy=True, batch_size=64, show_progress_bar=False))
    te_emb = torch.from_numpy(enc.encode(te_texts, convert_to_numpy=True, batch_size=64, show_progress_bar=False))

    print(f"train: {tuple(tr_emb.shape)}  test: {tuple(te_emb.shape)}")
    return feature_names, te_emb, te_labels, tr_emb, tr_labels


@app.cell
def _(mo):
    mo.md("""
    ## Training
    """)
    return


@app.cell
def _(
    HeadMoons,
    R_slider,
    alpha_slider,
    beta_slider,
    feature_names,
    k_slider,
    moon_feature,
    nn,
    np,
    predict_features,
    te_emb,
    te_labels,
    torch,
    tr_emb,
    tr_labels,
):
    import math as _math

    torch.manual_seed(0)
    np.random.seed(0)

    moon_idx = feature_names.index(moon_feature.value)
    predict_idxs = [feature_names.index(n) for n in predict_features.value]
    n_out = len(predict_idxs)
    alpha = alpha_slider.value
    beta = beta_slider.value
    h2_dim = k_slider.value

    model = HeadMoons(n_out=n_out, hidden2_dim=h2_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    tr_X, te_X = tr_emb, te_emb
    tr_y = torch.from_numpy(tr_labels)
    te_y = torch.from_numpy(te_labels)
    tr_y_pred = tr_y[:, predict_idxs]
    te_y_pred = te_y[:, predict_idxs]
    tr_y_moon = tr_y[:, moon_idx].long()

    n_train = tr_X.shape[0]
    batch_size = 128
    n_epochs = 40

    # Geometry — interleaving offset on center_1
    moon_R = R_slider.value
    moon_center_0 = torch.zeros(h2_dim)
    moon_center_1 = torch.zeros(h2_dim)
    moon_center_1[0]  = 1.0
    moon_center_1[-1] = 0.5 * moon_R
    moon_centers = torch.stack([moon_center_0, moon_center_1])

    # Direction targets — c_k along last axis
    c_k = _math.exp(_math.lgamma(h2_dim / 2) - 0.5 * _math.log(_math.pi) - _math.lgamma((h2_dim + 1) / 2))
    target_d_0 = torch.zeros(h2_dim); target_d_0[-1] = +c_k
    target_d_1 = torch.zeros(h2_dim); target_d_1[-1] = -c_k

    print(f"k={h2_dim}  c_k={c_k:.4f}  R={moon_R}")
    print(f"center_0 = {moon_center_0.tolist()}")
    print(f"center_1 = {moon_center_1.tolist()}")

    def shape_loss(z, y_moon):
        own_center = moon_centers[y_moon]
        z_centered = z - own_center
        norms = z_centered.norm(dim=1)
        L_radial = ((norms - moon_R) ** 2).mean()

        dirs = z_centered / (norms.unsqueeze(1) + 1e-6)
        mask_0 = (y_moon == 0).float().unsqueeze(1)
        mask_1 = 1.0 - mask_0
        n_0 = mask_0.sum().clamp(min=1.0)
        n_1 = mask_1.sum().clamp(min=1.0)
        mean_d0 = (dirs * mask_0).sum(dim=0) / n_0
        mean_d1 = (dirs * mask_1).sum(dim=0) / n_1

        L_dir = (mean_d0 - target_d_0).pow(2).sum() + (mean_d1 - target_d_1).pow(2).sum()
        return L_radial, L_dir

    hist_main = []
    hist_radial = []
    hist_dir = []
    for epoch in range(n_epochs):
        perm = torch.randperm(n_train)
        ep_main = 0.0; ep_radial = 0.0; ep_dir = 0.0
        n_batches = 0
        model.train()
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            x = tr_X[idx]; y = tr_y_pred[idx]; y_moon = tr_y_moon[idx]
            logits, z = model(x)
            L_main = bce(logits, y)
            L_radial_t, L_dir_t = shape_loss(z, y_moon)
            loss = L_main + alpha * L_radial_t + beta * L_dir_t
            opt.zero_grad(); loss.backward(); opt.step()
            ep_main += L_main.item()
            ep_radial += L_radial_t.item(); ep_dir += L_dir_t.item()
            n_batches += 1
        hist_main.append(ep_main / n_batches)
        hist_radial.append(ep_radial / n_batches)
        hist_dir.append(ep_dir / n_batches)

    print(f"Done.  k={h2_dim}  α={alpha}  β={beta}  R={moon_R}")
    print(f"Final L_main={hist_main[-1]:.3f}  L_radial={hist_radial[-1]:.3f}  L_dir={hist_dir[-1]:.4f}")

    return (
        hist_dir,
        hist_main,
        hist_radial,
        model,
        predict_idxs,
        te_X,
        te_y,
        te_y_pred,
        tr_X,
        tr_y_pred,
    )


@app.cell
def _(hist_dir, hist_main, hist_radial):
    import matplotlib.pyplot as plt

    fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
    ax_loss.plot(hist_main, label="L_main (BCE)")
    ax_loss.plot(hist_radial, label="L_radial")
    ax_loss.plot(hist_dir, label="L_dir")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("loss (unweighted)")
    ax_loss.set_yscale("log")
    ax_loss.legend()
    ax_loss.set_title("Training losses (log scale)")
    fig_loss.tight_layout()
    fig_loss

    return (plt,)


@app.cell
def _(mo):
    mo.md("""
    ## Evaluation
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Linear probe (on 2-D z) vs model's own head, per predicted feature
    """)
    return


@app.cell
def _(
    feature_names,
    model,
    predict_idxs,
    te_X,
    te_y_pred,
    torch,
    tr_X,
    tr_y_pred,
):
    from sklearn.linear_model import LogisticRegression
    import pandas as pd

    model.eval()
    with torch.no_grad():
        _, tr_z = model(tr_X); tr_z = tr_z.numpy()
        te_logits, te_z = model(te_X); te_logits = te_logits.numpy(); te_z = te_z.numpy()

    y_tr_int = tr_y_pred.numpy().astype(int)
    y_te_int = te_y_pred.numpy().astype(int)

    probe_accs = []
    own_accs = []
    for j, fi in enumerate(predict_idxs):
        lr = LogisticRegression(max_iter=2000)
        lr.fit(tr_z, y_tr_int[:, j])
        probe_accs.append(float(lr.score(te_z, y_te_int[:, j])))
        own_accs.append(float(((te_logits[:, j] > 0).astype(int) == y_te_int[:, j]).mean()))

    acc_df = pd.DataFrame({
        "feature": [feature_names[fi] for fi in predict_idxs],
        "probe acc on z (2-D)": probe_accs,
        "own-head acc": own_accs,
        "gap (head - probe)": [o - p for p, o in zip(probe_accs, own_accs)],
    }).round(3)
    acc_df
    return own_accs, probe_accs, te_z


@app.cell
def _(own_accs, plt, predict_features, probe_accs):
    import numpy as _np

    _names = predict_features.value
    _x = _np.arange(len(_names))
    _w = 0.4
    _cmap = plt.get_cmap("tab10")
    _colors = [_cmap(i) for i in range(len(_names))]

    fig_bar, ax_bar = plt.subplots(figsize=(max(5, 1.5 * len(_names)), 4))
    for _i, (_name, _p, _o, _c) in enumerate(zip(_names, probe_accs, own_accs, _colors)):
        ax_bar.bar(_x[_i] - _w / 2, _p, _w, color=_c, edgecolor="black", linewidth=0.5)
        ax_bar.bar(_x[_i] + _w / 2, _o, _w, color=_c, edgecolor="black", linewidth=0.5, hatch="///")

    ax_bar.set_xticks(_x)
    ax_bar.set_xticklabels(_names, rotation=20, ha="right")
    ax_bar.set_ylim(0, 1.05)
    ax_bar.axhline(0.5, linestyle="--", color="gray", linewidth=0.8)
    ax_bar.set_ylabel("test accuracy")
    ax_bar.set_title("Linear probe on z (solid) vs model's own head (hatched)")
    import matplotlib.patches as _mpatches
    _legend = [
        _mpatches.Patch(facecolor="lightgray", edgecolor="black", label="probe (linear on z)"),
        _mpatches.Patch(facecolor="lightgray", edgecolor="black", hatch="///", label="own head"),
    ]
    ax_bar.legend(handles=_legend, loc="lower right", fontsize=8)
    fig_bar.tight_layout()
    fig_bar
    return


@app.cell
def _(mo):
    mo.md("""
    ### 2-D scatter of hidden 2 — does the moon shape emerge?
    """)
    return


@app.cell
def _(R_slider, feature_names, moon_feature, np, plt, te_y, te_z):
    y_moon_te = te_y.numpy().astype(int)[:, feature_names.index(moon_feature.value)]
    _z_x = te_z[:, 0]
    _z_y = te_z[:, -1]
    _R = R_slider.value
    _c1_y = 0.5 * _R     # interleaving offset

    fig_scatter, axes = plt.subplots(1, 2, figsize=(11, 5))

    _theta = np.linspace(0, 2 * np.pi, 200)
    axes[0].plot(_R * np.cos(_theta), _R * np.sin(_theta), color="tab:blue", linewidth=2, label="class 0 circle")
    axes[0].plot(1 + _R * np.cos(_theta), _c1_y + _R * np.sin(_theta), color="tab:orange", linewidth=2, label="class 1 circle")
    axes[0].scatter([0, 1], [0, _c1_y], color="black", s=30, marker="x", label="centers")
    axes[0].set_aspect("equal")
    axes[0].set_title(f"Target: circles at (0,0) and (1, {_c1_y:.2f})  — R={_R:.2f}")
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel("z[0]"); axes[0].set_ylabel("z[-1]")

    axes[1].scatter(_z_x[y_moon_te == 0], _z_y[y_moon_te == 0], s=6, alpha=0.5, color="tab:blue", label=f"{moon_feature.value}=0")
    axes[1].scatter(_z_x[y_moon_te == 1], _z_y[y_moon_te == 1], s=6, alpha=0.5, color="tab:orange", label=f"{moon_feature.value}=1")
    axes[1].plot(_R * np.cos(_theta), _R * np.sin(_theta), color="tab:blue", linewidth=1, linestyle="--", alpha=0.4)
    axes[1].plot(1 + _R * np.cos(_theta), _c1_y + _R * np.sin(_theta), color="tab:orange", linewidth=1, linestyle="--", alpha=0.4)
    axes[1].scatter([0, 1], [0, _c1_y], color="black", s=30, marker="x")
    axes[1].set_aspect("equal")
    axes[1].set_title(f"Learned hidden 2 (projected to z[0] × z[-1])")
    axes[1].legend(fontsize=8)
    axes[1].set_xlabel("z[0]"); axes[1].set_ylabel("z[-1]")

    fig_scatter.tight_layout()
    fig_scatter

    return


if __name__ == "__main__":
    app.run()
