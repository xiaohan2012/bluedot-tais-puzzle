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
    # Q3 — L2-normalized hidden 2

    Force every hidden-2 activation onto the unit sphere by L2-normalizing
    after the third Linear layer (dropping the ReLU at hidden 2 so the
    vector can use the full sphere, not just the positive orthant).

    The radial degree of freedom is removed: only *direction* on the sphere
    can carry information. We study how this constraint affects the linear
    readability of each feature compared to the model's own head.
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
    k_slider = mo.ui.slider(
        start=2, stop=64, step=1, value=64,
        label="hidden-2 width k",
        show_value=True,
    )
    k_slider
    return (k_slider,)


@app.cell
def _(mo):
    mo.md("""
    ## Architecture
    """)
    return


@app.cell
def _(Fnn, nn):
    class HeadNorm(nn.Module):
        def __init__(self, hidden2_dim=64):
            super().__init__()
            self.h2_dim = hidden2_dim
            self.l1 = nn.Linear(384, 64)
            self.l2 = nn.Linear(64, 64)
            self.l3 = nn.Linear(64, hidden2_dim)   # hidden 2 — output L2-normalized below
            self.l4 = nn.Linear(hidden2_dim, 64)
            self.l5 = nn.Linear(64, 8)

        def hidden2(self, x):
            h = Fnn.relu(self.l1(x))
            h = Fnn.relu(self.l2(h))
            h = self.l3(h)                          # NO ReLU here — replaced by L2 norm
            return Fnn.normalize(h, dim=1)          # unit-norm: on the sphere

        def forward(self, x):
            h2 = self.hidden2(x)
            h = Fnn.relu(self.l4(h2))
            return self.l5(h)

    return (HeadNorm,)


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
    return feature_names, te_emb, te_labels, tr_emb, tr_labels


@app.cell
def _(mo):
    mo.md("""
    ## Training
    """)
    return


@app.cell
def _(HeadNorm, k_slider, nn, te_emb, te_labels, torch, tr_emb, tr_labels):
    torch.manual_seed(0)

    h2_dim = k_slider.value
    model = HeadNorm(hidden2_dim=h2_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    tr_X, te_X = tr_emb, te_emb
    tr_y = torch.from_numpy(tr_labels)
    te_y = torch.from_numpy(te_labels)

    n_epochs = 30
    batch_size = 128
    n_train = tr_X.shape[0]

    for epoch in range(n_epochs):
        perm = torch.randperm(n_train)
        ep_loss = 0.0
        n_batches = 0
        model.train()
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            logits = model(tr_X[idx])
            loss = bce(logits, tr_y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batches += 1

    # sanity: should all be ≈ 1
    with torch.no_grad():
        norms = model.hidden2(te_X[:200]).norm(dim=1)
    print(f"Done.  k={h2_dim}")
    print(f"Final main BCE = {ep_loss / n_batches:.3f}")
    print(f"||hidden_2|| on test sample: min={norms.min().item():.3f}  max={norms.max().item():.3f}  mean={norms.mean().item():.3f}")
    return h2_dim, model, te_X, te_y, tr_X, tr_y


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
def _(
    feature_names,
    h2_dim,
    model,
    np,
    own_accs,
    te_X,
    te_y,
    torch,
    tr_X,
    tr_y,
):
    from sklearn.linear_model import LogisticRegression
    import pandas as pd

    model.eval()
    with torch.no_grad():
        tr_h2 = model.hidden2(tr_X).numpy()
        te_h2 = model.hidden2(te_X).numpy()

    y_tr = tr_y.numpy().astype(int)
    y_te = te_y.numpy().astype(int)

    W = np.zeros((8, h2_dim))
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
    return W, probe_accs


@app.cell
def _(feature_names, own_accs, probe_accs):
    import matplotlib.pyplot as plt
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
    return (plt,)


@app.cell
def _(mo):
    mo.md("""
    ### Probe coefficient heatmap
    """)
    return


@app.cell
def _(W, feature_names, h2_dim, np, plt):
    Wabs = np.abs(W)
    Wn = Wabs / Wabs.max(axis=1, keepdims=True).clip(min=1e-12)
    fig_heat, ax_heat = plt.subplots(figsize=(max(6, h2_dim * 0.18), 4))
    im_heat = ax_heat.imshow(Wn, aspect="auto", cmap="viridis")
    ax_heat.set_yticks(range(8))
    ax_heat.set_yticklabels(feature_names)
    ax_heat.set_xlabel("hidden-2 neuron index")
    ax_heat.set_title(f"Linear probe |coefficient| per feature × neuron  (k={h2_dim}, rows max-normalized)")
    fig_heat.colorbar(im_heat, ax=ax_heat, label="|w| / row max")
    fig_heat.tight_layout()
    fig_heat
    return


@app.cell
def _(mo):
    mo.md("""
    ### Pairwise probe-direction cosines
    """)
    return


@app.cell
def _(W, feature_names, np, plt):
    Wnorm = W / np.linalg.norm(W, axis=1, keepdims=True).clip(min=1e-12)
    C = Wnorm @ Wnorm.T

    fig_cos, ax_cos = plt.subplots(figsize=(6, 5))
    im_cos = ax_cos.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
    ax_cos.set_xticks(range(8)); ax_cos.set_yticks(range(8))
    ax_cos.set_xticklabels(feature_names, rotation=45, ha="right")
    ax_cos.set_yticklabels(feature_names)
    for ci in range(8):
        for cj in range(8):
            ax_cos.text(cj, ci, f"{C[ci, cj]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(C[ci, cj]) > 0.5 else "black")
    ax_cos.set_title("Pairwise cosine similarity between probe weight vectors")
    fig_cos.colorbar(im_cos, ax=ax_cos, label="cos angle")
    fig_cos.tight_layout()
    fig_cos
    return


if __name__ == "__main__":
    app.run()
