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
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer, json, mo, nn, np, torch


@app.cell
def _(mo):
    mo.md("""
    # Q3 — Bottleneck on hidden 2

    Force the model to pack 8 features into `k < 8` hidden-2 dimensions
    so it cannot axis-align all of them. Some features end up dropped or
    non-linearly encoded — recoverable only via the downstream
    Linear→ReLU→Linear.

    Diagnostics: probe coefficient heatmap, pairwise cosine matrix between
    probe directions, and accuracy table (linear probe vs model's own head).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Configuration
    """)
    return


@app.cell
def _(mo):
    use_bottleneck = mo.ui.checkbox(value=True, label="Enable bottleneck")
    use_bottleneck
    return (use_bottleneck,)


@app.cell
def _(mo):
    k_slider = mo.ui.slider(
        start=2, stop=16, step=1, value=4,
        label="hidden-2 width k (when bottleneck on)",
        show_value=True,
    )
    k_slider
    return (k_slider,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Architecture
    """)
    return


@app.cell
def _(nn):
    class Head(nn.Module):
        def __init__(self, hidden2_dim=64):
            super().__init__()
            self.h2_dim = hidden2_dim
            self.layers = nn.Sequential(
                nn.Linear(384, 64),          nn.ReLU(),
                nn.Linear(64, 64),           nn.ReLU(),
                nn.Linear(64, hidden2_dim),  nn.ReLU(),   # hidden 2 \u2014 layer we inspect (post-ReLU)
                nn.Linear(hidden2_dim, 64),  nn.ReLU(),
                nn.Linear(64, 8),
            )

        def hidden2(self, x):
            return self.layers[:6](x)

        def forward(self, x):
            return self.layers(x)


    return (Head,)


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Training
    """)
    return


@app.cell
def _(
    Head,
    k_slider,
    nn,
    te_emb,
    te_labels,
    torch,
    tr_emb,
    tr_labels,
    use_bottleneck,
):
    torch.manual_seed(0)

    h2_dim = k_slider.value if use_bottleneck.value else 64

    model = Head(hidden2_dim=h2_dim)
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
            x = tr_X[idx]
            y = tr_y[idx]
            logits = model(x)
            loss = bce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batches += 1

    print(f"Done.  bottleneck={use_bottleneck.value} k={h2_dim}")
    print(f"Final main BCE = {ep_loss / n_batches:.3f}")
    return h2_dim, model, te_X, te_y, tr_X, tr_y


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Linear probes vs model’s own head
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
    }).round(3)
    acc_df
    return W, probe_accs, te_h2, y_te


@app.cell(hide_code=True)
def _(feature_names, own_accs, probe_accs):
    # Grouped bar chart: each feature one color; probe (solid) vs own-head (hatched).
    import matplotlib.pyplot as _plt
    import numpy as _np

    _x = _np.arange(len(feature_names))
    _w = 0.4
    _cmap = _plt.get_cmap("tab10")
    _colors = [_cmap(i) for i in range(len(feature_names))]

    fig_bar, ax_bar = _plt.subplots(figsize=(10, 4.5))
    for _i, (_name, _pacc, _oacc, _color) in enumerate(zip(feature_names, probe_accs, own_accs, _colors)):
        ax_bar.bar(_x[_i] - _w / 2, _pacc, _w, color=_color, edgecolor="black", linewidth=0.5)
        ax_bar.bar(_x[_i] + _w / 2, _oacc, _w, color=_color, edgecolor="black", linewidth=0.5, hatch="///")

    ax_bar.set_xticks(_x)
    ax_bar.set_xticklabels(feature_names, rotation=30, ha="right")
    ax_bar.set_ylim(0, 1.05)
    ax_bar.axhline(0.5, linestyle="--", color="gray", linewidth=0.8, label="chance")
    ax_bar.set_ylabel("test accuracy")
    ax_bar.set_title("Linear probe (solid) vs model\u2019s own head (hatched) per feature")
    # Pattern legend (one entry per pattern, color-agnostic)
    import matplotlib.patches as _mpatches
    _legend = [
        _mpatches.Patch(facecolor="lightgray", edgecolor="black", label="probe"),
        _mpatches.Patch(facecolor="lightgray", edgecolor="black", hatch="///", label="own head"),
    ]
    ax_bar.legend(handles=_legend, loc="lower right")
    fig_bar.tight_layout()
    fig_bar
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Probe coefficient heatmap
    """)
    return


@app.cell
def _(W, feature_names, h2_dim, np):
    import matplotlib.pyplot as plt

    Wabs = np.abs(W)
    Wn = Wabs / Wabs.max(axis=1, keepdims=True).clip(min=1e-12)
    fig_heat, ax_heat = plt.subplots(figsize=(max(6, h2_dim * 0.7), 4))
    im_heat = ax_heat.imshow(Wn, aspect="auto", cmap="viridis")
    ax_heat.set_yticks(range(8))
    ax_heat.set_yticklabels(feature_names)
    ax_heat.set_xticks(range(h2_dim))
    ax_heat.set_xlabel("hidden-2 neuron index")
    ax_heat.set_title(f"Linear probe |coefficient| per feature × neuron  (h2_dim={h2_dim}, rows max-normalized)")
    fig_heat.colorbar(im_heat, ax=ax_heat, label="|w| / row max")
    fig_heat.tight_layout()
    fig_heat
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Pairwise probe-direction cosines
    """)
    return


@app.cell
def _(W, feature_names, np, plt):
    # Cosine similarity between probe weight vectors (8x8). Off-diagonal magnitudes >> 0 ⇒ superposition.
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
    ax_cos.set_title("Pairwise cosine similarity between probe weight vectors\n(off-diagonal ≫ 0 ⇒ non-orthogonal ⇒ superposition)")
    fig_cos.colorbar(im_cos, ax=ax_cos, label="cos angle")
    fig_cos.tight_layout()
    fig_cos
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### PCA of hidden 2 — colored by each feature
    """)
    return


@app.cell(hide_code=True)
def _(feature_names, mo):
    pca_feature_select = mo.ui.multiselect(
        options=list(feature_names),
        value=list(feature_names),
        label="Features to plot",
    )
    pca_feature_select
    return (pca_feature_select,)


@app.cell(hide_code=True)
def _(feature_names, h2_dim, pca_feature_select, plt, te_h2, y_te):
    # 2-D PCA of hidden 2, one subplot per selected feature
    from sklearn.decomposition import PCA as _PCA

    _pca = _PCA(n_components=2, random_state=0)
    _xy = _pca.fit_transform(te_h2)
    _evr = _pca.explained_variance_ratio_

    _selected = pca_feature_select.value or list(feature_names)
    _n = len(_selected)
    _ncols = min(4, _n)
    _nrows = max(1, (_n + _ncols - 1) // _ncols)

    fig_pca_all, axes_pca_all = plt.subplots(
        _nrows, _ncols, figsize=(3.5 * _ncols, 3.5 * _nrows),
        sharex=True, sharey=True, squeeze=False,
    )
    _flat = axes_pca_all.flat
    for _ax_pca, _name in zip(_flat, _selected):
        _fi = feature_names.index(_name)
        _y = y_te[:, _fi]
        _ax_pca.scatter(_xy[_y == 0, 0], _xy[_y == 0, 1], s=4, alpha=0.35, color="tab:blue",   label="0")
        _ax_pca.scatter(_xy[_y == 1, 0], _xy[_y == 1, 1], s=4, alpha=0.35, color="tab:orange", label="1")
        _ax_pca.set_title(_name)
        _ax_pca.set_xticks([]); _ax_pca.set_yticks([])
    for _ax_pca in list(_flat)[_n:]:
        _ax_pca.axis("off")
    fig_pca_all.suptitle(f"PCA of hidden 2 (k={h2_dim}) — PC1={_evr[0]:.2f}, PC2={_evr[1]:.2f}", y=1.00)
    fig_pca_all.tight_layout()
    fig_pca_all
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## k-sweep — how many features are encoded linearly vs non-linearly vs not at all

    For each `k`, train a fresh model with `hidden2_dim=k` (seed=0). Classify each of the 8 features into one of three buckets at threshold τ = 0.85:

    - **both**: probe acc ≥ τ **and** own-head acc ≥ τ — encoded linearly at hidden 2
    - **head only**: own-head ≥ τ but probe < τ — head recovers it, but not from a linear direction (non-linear / superposed encoding)
    - **neither**: own-head < τ — the model failed to learn this feature at this `k`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    run_sweep_btn = mo.ui.run_button(label="Run k-sweep")
    run_sweep_btn
    return (run_sweep_btn,)


@app.cell(hide_code=True)
def _(
    Head,
    feature_names,
    mo,
    nn,
    np,
    run_sweep_btn,
    te_emb,
    te_labels,
    torch,
    tr_emb,
    tr_labels,
):
    # k-sweep: train, probe, classify per feature. Triggered by run_sweep_btn.
    mo.stop(not run_sweep_btn.value, mo.md("Press the button to run the sweep."))

    from sklearn.linear_model import LogisticRegression as _LR_s
    import pandas as _pd_s

    K_VALUES = list(range(2, 17))
    TAU = 0.85

    def _train_and_eval(k, seed=0, n_epochs=30, batch_size=128, lr=1e-3):
        torch.manual_seed(seed)
        m = Head(hidden2_dim=k)
        opt_ = torch.optim.Adam(m.parameters(), lr=lr)
        bce_ = nn.BCEWithLogitsLoss()
        n = tr_emb.shape[0]
        tr_y_ = torch.from_numpy(tr_labels)
        for _ in range(n_epochs):
            perm = torch.randperm(n)
            m.train()
            for s in range(0, n, batch_size):
                idx = perm[s:s+batch_size]
                logits = m(tr_emb[idx])
                loss = bce_(logits, tr_y_[idx])
                opt_.zero_grad(); loss.backward(); opt_.step()
        m.eval()
        with torch.no_grad():
            tr_h2_ = m.hidden2(tr_emb).numpy()
            te_h2_ = m.hidden2(te_emb).numpy()
            te_logits_ = m(te_emb).numpy()
        y_tr_ = tr_labels.astype(int)
        y_te_ = te_labels.astype(int)
        probe = []
        head  = []
        for _fi in range(8):
            lr_ = _LR_s(max_iter=2000).fit(tr_h2_, y_tr_[:, _fi])
            probe.append(float(lr_.score(te_h2_, y_te_[:, _fi])))
            head.append(float(((te_logits_[:, _fi] > 0).astype(int) == y_te_[:, _fi]).mean()))
        return np.array(probe), np.array(head)

    _rows = []
    _per_feature_rows = []
    for _k in K_VALUES:
        print(f"k={_k:>2} ...", end=" ", flush=True)
        _p, _h = _train_and_eval(_k)
        _both  = int(((_p >= TAU) & (_h >= TAU)).sum())
        _head_only = int(((_p < TAU) & (_h >= TAU)).sum())
        _neither = int((_h < TAU).sum())
        _rows.append({"k": _k, "both": _both, "head_only": _head_only, "neither": _neither})
        for _fi, _fname in enumerate(feature_names):
            _per_feature_rows.append({"k": _k, "feature": _fname,
                                       "probe": _p[_fi], "head": _h[_fi]})
        print(f"both={_both} head_only={_head_only} neither={_neither}")

    sweep_df = _pd_s.DataFrame(_rows)
    sweep_per_feature_df = _pd_s.DataFrame(_per_feature_rows)
    sweep_df
    return (sweep_df,)


@app.cell(hide_code=True)
def _(sweep_df):
    # Stacked bar of the sweep
    import matplotlib.pyplot as _plt_s

    fig_sw, _ax = _plt_s.subplots(figsize=(9, 4.5))
    _x = sweep_df["k"].values
    _ax.bar(_x, sweep_df["both"],     color="tab:green",  label="both (linear at hidden 2)")
    _ax.bar(_x, sweep_df["head_only"], bottom=sweep_df["both"],
            color="tab:orange", label="head only (non-linear)")
    _ax.bar(_x, sweep_df["neither"],
            bottom=sweep_df["both"] + sweep_df["head_only"],
            color="tab:red", label="neither (not learned)")
    _ax.set_xlabel("hidden-2 width k")
    _ax.set_ylabel("# features (out of 8)")
    _ax.set_title("Feature classification across k  (τ=0.85)")
    _ax.set_xticks(_x)
    _ax.set_ylim(0, 8.5)
    _ax.legend(fontsize=9, loc="lower right")
    fig_sw.tight_layout()
    fig_sw
    return


if __name__ == "__main__":
    app.run()
