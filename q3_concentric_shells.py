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
    # Q3 — Concentric shells

    Encode one feature F by **radius** of the hidden-2 activation, not by
    direction. Train with two auxiliary losses on hidden 2:

    - **L_radial** — pulls F=1 samples toward target radius `R1` and F=0
      samples toward `R0`.
    - **L_iso** — makes the *direction* of hidden 2 class-independent for F,
      so no linear projection can recover F.

    Expected outcome: linear probes on F drop toward chance (no direction
    carries F); the model's own head still recovers F by reading the norm
    through `Linear → ReLU → Linear`. The other 7 features stay in the
    angular subspace and remain linearly readable.

    Architecture note: we drop the ReLU at hidden 2 so the vector can be
    signed (centered around 0), as required for the "concentric about origin"
    geometry.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Configuration
    """)
    return


app._unparsable_cell(
    r"""
     = mo.ui.dropdown(
        options=["number", "question", "color", "food", "sentiment", "country", "person", "body_part"],
        value="sentiment",
        label="Feature F to encode by radius",
    )

    """,
    name="_"
)


@app.cell
def _(mo):
    alpha_slider = mo.ui.slider(
        start=0.0, stop=2.0, step=0.05, value=0.5,
        label="α — radial loss weight",
        show_value=True,
    )
    alpha_slider
    return


@app.cell
def _(mo):
    beta_slider = mo.ui.slider(
        start=0.0, stop=2.0, step=0.05, value=0.5,
        label="β — direction-isotropy loss weight",
        show_value=True,
    )
    beta_slider
    return


@app.cell
def _(mo):
    r0_slider = mo.ui.slider(start=0.5, stop=3.0, step=0.1, value=1.0, label="R₀ (radius for F=0)", show_value=True)
    r1_slider = mo.ui.slider(start=2.0, stop=8.0, step=0.1, value=4.0, label="R₁ (radius for F=1)", show_value=True)
    mo.hstack([r0_slider, r1_slider])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Architecture
    """)
    return


@app.cell
def _(Fnn, nn):
    class HeadShell(nn.Module):
        # Hidden 2 is the OUTPUT of the third Linear, with NO ReLU — so it can
        # take any sign and live anywhere in R^64. This is what lets the
        # concentric-shell geometry exist.
        def __init__(self, hidden2_dim=64):
            super().__init__()
            self.h2_dim = hidden2_dim
            self.l1 = nn.Linear(384, 64)
            self.l2 = nn.Linear(64, 64)
            self.l3 = nn.Linear(64, hidden2_dim)   # hidden 2 — signed, no ReLU
            self.l4 = nn.Linear(hidden2_dim, 64)
            self.l5 = nn.Linear(64, 8)

        def hidden2(self, x):
            h = Fnn.relu(self.l1(x))
            h = Fnn.relu(self.l2(h))
            return self.l3(h)                       # signed; no nonlinearity

        def forward(self, x):
            h2 = self.hidden2(x)
            h = Fnn.relu(self.l4(h2))
            return self.l5(h)

    return


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

    print(f"train embeddings: {tuple(tr_emb.shape)}   test: {tuple(te_emb.shape)}")
    return (feature_names,)


@app.cell
def _(mo):
    mo.md("""
    ## Training
    """)
    return


app._unparsable_cell(
    r"""
    torch.manual_seed(0)

    f_idx = feature_names.index(.value)
    R0 = r0_slider.value
    R1 = r1_slider.value
    alpha = alpha_slider.value
    beta = beta_slider.value

    model = HeadShell(hidden2_dim=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    tr_X, te_X = tr_emb, te_emb
    tr_y = torch.from_numpy(tr_labels)
    te_y = torch.from_numpy(te_labels)
    tr_y_F = tr_y[:, f_idx]

    n_epochs = 40
    batch_size = 128
    n_train = tr_X.shape[0]

    hist_main = []
    hist_rad = []
    hist_iso = []
    for epoch in range(n_epochs):
        perm = torch.randperm(n_train)
        ep_main = 0.0; ep_rad = 0.0; ep_iso = 0.0; n_batches = 0
        model.train()
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            x = tr_X[idx]
            y_all = tr_y[idx]
            y_F = tr_y_F[idx]

            h2 = model.hidden2(x)                       # (B, 64), signed
            h2c = h2 - h2.mean(dim=0, keepdim=True)     # batch-centered
            norms = h2c.norm(dim=1)                     # (B,)
            target_R = torch.where(y_F == 1, torch.tensor(R1), torch.tensor(R0))
            L_radial = ((norms - target_R) ** 2).mean()

            dirs = h2c / (norms.unsqueeze(1) + 1e-6)
            # Class-conditional mean directions; iso loss = squared dist between them
            mask_pos = (y_F == 1).float().unsqueeze(1)
            mask_neg = 1.0 - mask_pos
            n_pos = mask_pos.sum().clamp(min=1.0)
            n_neg = mask_neg.sum().clamp(min=1.0)
            mean_pos = (dirs * mask_pos).sum(dim=0) / n_pos
            mean_neg = (dirs * mask_neg).sum(dim=0) / n_neg
            L_iso = (mean_pos - mean_neg).pow(2).sum()

            # Main task: standard 8-way BCE; forward uses uncentered h2
            logits = model.l5(torch.relu(model.l4(h2)))
            L_main = bce(logits, y_all)

            loss = L_main + alpha * L_radial + beta * L_iso
            opt.zero_grad(); loss.backward(); opt.step()

            ep_main += L_main.item(); ep_rad += L_radial.item(); ep_iso += L_iso.item(); n_batches += 1

        hist_main.append(ep_main / n_batches)
        hist_rad.append(ep_rad / n_batches)
        hist_iso.append(ep_iso / n_batches)

    print(f"Done.  F={.value}  α={alpha}  β={beta}  R0={R0}  R1={R1}")
    print(f"Final L_main = {hist_main[-1]:.3f}   L_radial = {hist_rad[-1]:.3f}   L_iso = {hist_iso[-1]:.4f}")
    """,
    name="_"
)


@app.cell
def _(hist_iso, hist_main, hist_rad):
    import matplotlib.pyplot as plt

    fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
    ax_loss.plot(hist_main, label="L_main (BCE)")
    ax_loss.plot(hist_rad, label="L_radial")
    ax_loss.plot(hist_iso, label="L_iso")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
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
def _(feature_names, model, te_X, te_y, torch):
    model.eval()
    with torch.no_grad():
        te_logits = model(te_X).numpy()
    _te_y_np = te_y.numpy().astype(int)
    own_accs = []
    for _i in range(8):
        _pred = (te_logits[:, _i] > 0).astype(int)
        own_accs.append(float((_pred == _te_y_np[:, _i]).mean()))
    print("Model's own head accuracy per feature:")
    for _n, _a in zip(feature_names, own_accs):
        print(f"  {_n:<12} {_a:.3f}")
    return (own_accs,)


@app.cell
def _(mo):
    mo.md("""
    ### Linear probes vs model's own head
    """)
    return


@app.cell
def _(feature_names, model, np, own_accs, te_X, te_y, torch, tr_X, tr_y):
    from sklearn.linear_model import LogisticRegression
    import pandas as pd

    model.eval()
    with torch.no_grad():
        tr_h2 = model.hidden2(tr_X).numpy()
        te_h2 = model.hidden2(te_X).numpy()

    y_tr = tr_y.numpy().astype(int)
    y_te = te_y.numpy().astype(int)

    W = np.zeros((8, tr_h2.shape[1]))
    probe_accs = []
    for fi, fname in enumerate(feature_names):
        lr = LogisticRegression(max_iter=2000)
        lr.fit(tr_h2, y_tr[:, fi])
        W[fi] = lr.coef_.ravel()
        probe_accs.append(float(lr.score(te_h2, y_te[:, fi])))

    acc_df = pd.DataFrame({
        "feature": feature_names,
        "probe acc (linear)": probe_accs,
        "own-head acc": own_accs,
        "gap (head - probe)": [o - p for p, o in zip(probe_accs, own_accs)],
    }).round(3)
    acc_df
    return (probe_accs,)


@app.cell
def _(feature_names, own_accs, plt, probe_accs):
    import numpy as _np

    _x = _np.arange(len(feature_names))
    _w = 0.4
    _cmap = plt.get_cmap("tab10")
    _colors = [_cmap(i) for i in range(len(feature_names))]

    fig_bar, ax_bar = plt.subplots(figsize=(10, 4.5))
    for _i, (_name, _pacc, _oacc, _color) in enumerate(zip(feature_names, probe_accs, own_accs, _colors)):
        ax_bar.bar(_x[_i] - _w / 2, _pacc, _w, color=_color, edgecolor="black", linewidth=0.5)
        ax_bar.bar(_x[_i] + _w / 2, _oacc, _w, color=_color, edgecolor="black", linewidth=0.5, hatch="///")

    ax_bar.set_xticks(_x)
    ax_bar.set_xticklabels(feature_names, rotation=30, ha="right")
    ax_bar.set_ylim(0, 1.05)
    ax_bar.axhline(0.5, linestyle="--", color="gray", linewidth=0.8)
    ax_bar.set_ylabel("test accuracy")
    ax_bar.set_title("Linear probe (solid) vs model's own head (hatched) per feature")
    import matplotlib.patches as _mpatches
    _legend = [
        _mpatches.Patch(facecolor="lightgray", edgecolor="black", label="probe"),
        _mpatches.Patch(facecolor="lightgray", edgecolor="black", hatch="///", label="own head"),
    ]
    ax_bar.legend(handles=_legend, loc="lower right")
    fig_bar.tight_layout()
    fig_bar
    return


@app.cell
def _(mo):
    mo.md("""
    ### Activation norm distribution — the shell diagnostic
    """)
    return


app._unparsable_cell(
    r"""
    _y_F_te = te_y.numpy().astype(int)[:, f_idx]
    _h2_centered = te_h2 - te_h2.mean(axis=0, keepdims=True)
    _norms = np.linalg.norm(_h2_centered, axis=1)

    fig_norm, ax_norm = plt.subplots(figsize=(8, 4))
    ax_norm.hist(_norms[_y_F_te == 0], bins=50, alpha=0.5, density=True, label=f"{.value}=0", color="tab:blue")
    ax_norm.hist(_norms[_y_F_te == 1], bins=50, alpha=0.5, density=True, label=f"{.value}=1", color="tab:orange")
    ax_norm.axvline(r0_slider.value, linestyle="--", color="tab:blue", alpha=0.6, label=f"R₀={r0_slider.value:.1f}")
    ax_norm.axvline(r1_slider.value, linestyle="--", color="tab:orange", alpha=0.6, label=f"R₁={r1_slider.value:.1f}")
    ax_norm.set_xlabel("‖h₂ − mean(h₂)‖ on test")
    ax_norm.set_ylabel("density")
    ax_norm.set_title(f"Hidden-2 norm distribution split by {.value}")
    ax_norm.legend()
    fig_norm.tight_layout()
    fig_norm
    """,
    name="_"
)


@app.cell
def _(mo):
    mo.md("""
    ### Norm as a single-feature probe
    """)
    return


app._unparsable_cell(
    r"""
    # Can a 1-feature linear probe (using only ‖h‖) recover F?
    from sklearn.linear_model import LogisticRegression as _LR

    tr_centered = tr_h2 - tr_h2.mean(axis=0, keepdims=True)
    te_centered = te_h2 - te_h2.mean(axis=0, keepdims=True)
    n_tr = np.linalg.norm(tr_centered, axis=1).reshape(-1, 1)
    n_te = np.linalg.norm(te_centered, axis=1).reshape(-1, 1)

    y_tr_F = tr_y.numpy().astype(int)[:, f_idx]
    y_te_F = te_y.numpy().astype(int)[:, f_idx]

    lr_norm = _LR(max_iter=1000)
    lr_norm.fit(n_tr, y_tr_F)
    acc_norm_probe = lr_norm.score(n_te, y_te_F)

    print(f"1-feature probe (just ‖h‖) on {.value}: test acc = {acc_norm_probe:.3f}")
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
