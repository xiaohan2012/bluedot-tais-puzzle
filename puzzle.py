import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import torch
    import torch.nn as nn
    from sentence_transformers import SentenceTransformer
    import marimo as mo


    return SentenceTransformer, mo, nn, torch


@app.cell
def _(nn):
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(384, 64), nn.ReLU(),   # hidden 0
                nn.Linear(64, 64),  nn.ReLU(),   # hidden 1
                nn.Linear(64, 64),  nn.ReLU(),   # hidden 2  ← non-linear activation here (post-ReLU)
                nn.Linear(64, 64),  nn.ReLU(),   # hidden 3
                nn.Linear(64, 8),                # logits
            )

        def forward(self, x):
            return self.layers(x)

    return (Head,)


@app.cell
def _(Head, SentenceTransformer, torch):
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    m = Head()
    m.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=False))
    m.eval()
    return enc, m


@app.cell
def _(enc, m, torch):
    texts = [
        "Alice loves the red car she bought in Japan for two hundred dollars.",
        "Did Sarah eat pizza with her hands in Italy?",
    ]

    with torch.no_grad():
        embeddings = torch.from_numpy(
            enc.encode(texts, convert_to_numpy=True)
        )
        logits = m(embeddings)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
    return embeddings, preds, texts


@app.cell
def _(embeddings, m, torch):
    with torch.no_grad():
        layer2_acts = m.layers[:6](embeddings)
    return (layer2_acts,)


@app.cell
def _(layer2_acts, preds, texts):
    import json

    feature_names = json.load(open("feature_names.json"))

    for text, p in zip(texts, preds):
        active = [name for name, v in zip(feature_names, p.tolist()) if v == 1]
        print(f"  {text}")
        print(f"    -> {active}")

    print(f"\nlayer 2 activations: {tuple(layer2_acts.shape)}")
    return


@app.cell(hide_code=True)
def _(mo):
    # Data source selector
    source_choice = mo.ui.dropdown(
        options={"train (data/train.jsonl)": "data/train.jsonl",
                 "test  (data/test.jsonl)":  "data/test.jsonl"},
        value="train (data/train.jsonl)",
        label="Input source",
    )
    source_choice
    return (source_choice,)


@app.cell(hide_code=True)
def _(mo):
    # Hidden layer to probe (post-ReLU)
    layer_choice = mo.ui.dropdown(
        options={"hidden 0 (post-ReLU)": 2,
                 "hidden 1 (post-ReLU)": 4,
                 "hidden 2 (post-ReLU)": 6,
                 "hidden 3 (post-ReLU)": 8},
        value="hidden 2 (post-ReLU)",
        label="Activation layer",
    )
    layer_choice
    return (layer_choice,)


@app.cell(hide_code=True)
def _(source_choice):
    # Load BOTH train and test always (train is fit set, test is generalization set).
    # source_choice still drives what's shown in the histogram / PCA plots.
    import json as _json

    with open("data/train.jsonl") as _f:
        _tr_rows = [_json.loads(line) for line in _f]
    with open("data/test.jsonl") as _f:
        _te_rows = [_json.loads(line) for line in _f]

    tr_texts = [r["text"] for r in _tr_rows]
    tr_labels = [r["labels"] for r in _tr_rows]
    te_texts = [r["text"] for r in _te_rows]
    te_labels = [r["labels"] for r in _te_rows]

    if source_choice.value == "data/train.jsonl":
        train_texts, train_labels = tr_texts, tr_labels
    else:
        train_texts, train_labels = te_texts, te_labels

    feature_names_full = _json.load(open("feature_names.json"))
    print(f"train(fit)={len(tr_texts)}  test(eval)={len(te_texts)}  viz={len(train_texts)} from {source_choice.value}")
    return feature_names_full, te_labels, te_texts, tr_labels, tr_texts


@app.cell(hide_code=True)
def _(
    enc,
    layer_choice,
    m,
    source_choice,
    te_labels,
    te_texts,
    torch,
    tr_labels,
    tr_texts,
):
    # Encode all splits and extract activations at the chosen layer
    import numpy as np

    with torch.no_grad():
        _e_tr = torch.from_numpy(enc.encode(tr_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64))
        _e_te = torch.from_numpy(enc.encode(te_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64))
        tr_acts = m.layers[:layer_choice.value](_e_tr).numpy()
        te_acts = m.layers[:layer_choice.value](_e_te).numpy()

    tr_labels_arr = np.array(tr_labels)
    te_labels_arr = np.array(te_labels)

    # Visualization data tracks source_choice
    if source_choice.value == "data/train.jsonl":
        train_acts, train_labels_arr = tr_acts, tr_labels_arr
    else:
        train_acts, train_labels_arr = te_acts, te_labels_arr

    print(f"layer slice [:{layer_choice.value}]  tr_acts: {tr_acts.shape}  te_acts: {te_acts.shape}  viz: {train_acts.shape}")
    return (
        np,
        te_acts,
        te_labels_arr,
        tr_acts,
        tr_labels_arr,
        train_acts,
        train_labels_arr,
    )


@app.cell(hide_code=True)
def _(mo):
    # Classifier for the COUNTRY probe
    clf_choice = mo.ui.dropdown(
        options=[
            "Logistic Regression",
            "Degree-2 polynomial LogReg",
            "MLP (1 hidden, 32 units)",
        ],
        value="MLP (1 hidden, 32 units)",
        label="Country probe classifier",
    )
    clf_choice
    return (clf_choice,)


@app.cell(hide_code=True)
def _(feature_names_full, mo):
    # A non-country feature to compare against (always probed with Logistic Regression)
    _others = [n for n in feature_names_full if n != "country"]
    compare_choice = mo.ui.dropdown(
        options=_others,
        value="person",
        label="Comparison feature (LogReg)",
    )
    compare_choice
    return (compare_choice,)


@app.cell(hide_code=True)
def _(
    clf_choice,
    compare_choice,
    feature_names_full,
    np,
    te_acts,
    te_labels_arr,
    tr_acts,
    tr_labels_arr,
    train_acts,
    train_labels_arr,
):
    # Two probes only:
    #   - country, using clf_choice
    #   - compare_choice.value, using Logistic Regression
    # Fit on train, evaluate on test, score the visualization set.
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    def _make_clf(name):
        if name == "Logistic Regression":
            return LogisticRegression(max_iter=1000)
        if name == "Degree-2 polynomial LogReg":
            return make_pipeline(
                PolynomialFeatures(degree=2, include_bias=False),
                LogisticRegression(max_iter=2000, C=0.1),
            )
        if name == "MLP (1 hidden, 32 units)":
            return MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=500, random_state=0)
        raise ValueError(name)

    def _score(clf, X):
        if hasattr(clf, "decision_function"):
            return clf.decision_function(X)
        lp = clf.predict_log_proba(X)
        s = lp[:, 1] - lp[:, 0]
        return np.nan_to_num(s, nan=0.0, posinf=20.0, neginf=-20.0)

    probe_panels = []
    for _spec in [("country", clf_choice.value), (compare_choice.value, "Logistic Regression")]:
        _name, _clf_name = _spec
        _i = feature_names_full.index(_name)
        _clf = _make_clf(_clf_name)
        _clf.fit(tr_acts, tr_labels_arr[:, _i])
        probe_panels.append({
            "name": _name,
            "clf": _clf_name,
            "scores": _score(_clf, train_acts),
            "y_viz": train_labels_arr[:, _i],
            "train_acc": _clf.score(tr_acts, tr_labels_arr[:, _i]),
            "test_acc": _clf.score(te_acts, te_labels_arr[:, _i]),
        })

    print(f"{'feature':<12} {'classifier':<30} {'train':>8} {'test':>8}")
    for _p in probe_panels:
        print(f"  {_p['name']:<10} {_p['clf']:<30} {_p['train_acc']:>8.3f} {_p['test_acc']:>8.3f}")
    return LogisticRegression, probe_panels


@app.cell(hide_code=True)
def _(probe_panels):
    # Histograms of probe scores: country (left) vs comparison feature (right)
    import matplotlib.pyplot as plt

    fig_hist, axes_hist = plt.subplots(1, 2, figsize=(11, 4))
    for _ax, _p in zip(axes_hist, probe_panels):
        _s, _y = _p["scores"], _p["y_viz"]
        _ax.hist(_s[_y == 0], bins=50, alpha=0.5, density=True, label="0", color="tab:blue")
        _ax.hist(_s[_y == 1], bins=50, alpha=0.5, density=True, label="1", color="tab:orange")
        _ax.set_title(f"{_p['name']} — {_p['clf']}\ntrain={_p['train_acc']:.2f}  test={_p['test_acc']:.2f}")
        _ax.set_xlabel("probe score (log-odds)")
        _ax.set_ylabel("density")
        _ax.legend(fontsize=9)
    fig_hist.tight_layout()
    fig_hist
    return (plt,)


@app.cell(hide_code=True)
def _(train_acts):
    # PCA on the 7000 × 64 activations (one global 2-D embedding)
    from sklearn.decomposition import PCA

    _pca = PCA(n_components=2, random_state=0)
    pca_xy = _pca.fit_transform(train_acts)
    print("PCA done:", pca_xy.shape, "explained var ratio:", _pca.explained_variance_ratio_)
    return (pca_xy,)


@app.cell(hide_code=True)
def _(pca_xy, plt, probe_panels):
    # PCA scatter: country (left) vs comparison feature (right)
    fig_pca, axes_pca = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for _ax, _p in zip(axes_pca, probe_panels):
        _y = _p["y_viz"]
        _ax.scatter(pca_xy[_y == 0, 0], pca_xy[_y == 0, 1], s=4, alpha=0.4, color="tab:blue", label="0")
        _ax.scatter(pca_xy[_y == 1, 0], pca_xy[_y == 1, 1], s=4, alpha=0.4, color="tab:orange", label="1")
        _ax.set_title(_p["name"])
        _ax.set_xlabel("PC1"); _ax.set_ylabel("PC2")
        _ax.legend(fontsize=9, markerscale=2)
    fig_pca.suptitle("PCA of layer-2 activations, colored by label", y=1.02)
    fig_pca.tight_layout()
    fig_pca
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Encoding of the 7 linear features — probe coefficient heatmap
    """)
    return


@app.cell(hide_code=True)
def _(LogisticRegression, feature_names_full, np, plt, tr_acts, tr_labels_arr):
    # Probe-coefficient heatmap: |w_f,k| for the 7 linearly-separable features.
    # Rows = features (excluding country), cols = 64 hidden-2 neurons.
    # Each row is normalized by its max so patterns are comparable across features.
    _linear_feats = [n for n in feature_names_full if n != "country"]

    _W = np.zeros((len(_linear_feats), tr_acts.shape[1]))
    for _row, _name in enumerate(_linear_feats):
        _i = feature_names_full.index(_name)
        _lr = LogisticRegression(max_iter=1000)
        _lr.fit(tr_acts, tr_labels_arr[:, _i])
        _W[_row] = np.abs(_lr.coef_.ravel())

    _Wn = _W / _W.max(axis=1, keepdims=True)  # per-row max-normalized

    fig_coef, _ax = plt.subplots(figsize=(14, 4))
    _im = _ax.imshow(_Wn, aspect="auto", cmap="viridis")
    _ax.set_yticks(range(len(_linear_feats)))
    _ax.set_yticklabels(_linear_feats)
    _ax.set_xticks(range(0, _W.shape[1], 4))
    _ax.set_xlabel("hidden-2 neuron index")
    _ax.set_title("Linear probe |coefficient| per feature × neuron  (each row normalized by its max)")
    fig_coef.colorbar(_im, ax=_ax, label="|w| / max(|w|) per feature")
    fig_coef.tight_layout()

    print("Top-5 neurons by |coef| per feature:")
    for _row, _name in enumerate(_linear_feats):
        _top = np.argsort(-_W[_row])[:5]
        _frac = _W[_row, _top].sum() / _W[_row].sum()
        print(f"  {_name:<10} top-5 neurons = {_top.tolist()}   share of |w|_1 = {_frac:.2f}")

    fig_coef
    return


@app.cell(hide_code=True)
def _(
    feature_names_full,
    plt,
    tr_acts,
    tr_labels_arr,
    train_acts,
    train_labels_arr,
):
    # Log-odds distribution on country: degree-2 polynomial vs linear LogReg
    from sklearn.linear_model import LogisticRegression as _LR2
    from sklearn.preprocessing import PolynomialFeatures as _Poly
    from sklearn.pipeline import make_pipeline as _mkpipe
    import numpy as _np2

    _y_tr = tr_labels_arr[:, feature_names_full.index("country")]
    _y_viz = train_labels_arr[:, feature_names_full.index("country")]

    _lin = _LR2(max_iter=1000).fit(tr_acts, _y_tr)
    _poly = _mkpipe(_Poly(degree=2, include_bias=False), _LR2(max_iter=2000, C=0.1)).fit(tr_acts, _y_tr)

    _lo_lin = _lin.decision_function(train_acts)
    _lo_poly = _poly.decision_function(train_acts)

    fig_lo, axes_lo = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for _ax, _lo, _name in [
        (axes_lo[0], _lo_poly, "Degree-2 poly LogReg"),
        (axes_lo[1], _lo_lin,  "Linear LogReg"),
    ]:
        _bins = _np2.linspace(_lo.min(), _lo.max(), 60)
        _ax.hist(_lo[_y_viz == 0], bins=_bins, alpha=0.5, color="tab:blue",   label="country=0")
        _ax.hist(_lo[_y_viz == 1], bins=_bins, alpha=0.5, color="tab:orange", label="country=1")
        _ax.axvline(0, color="k", lw=0.8, ls="--")
        _ax.set_title(_name)
        _ax.set_xlabel("log-odds")
        _ax.legend(fontsize=9)
    axes_lo[0].set_ylabel("count")
    fig_lo.suptitle("Country log-odds distribution at hidden 2", y=1.02)
    fig_lo.tight_layout()
    fig_lo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Per-neuron activation statistics at hidden 2

    For each of the 64 post-ReLU neurons:  mean activation, std, fraction of inputs where it fires (>0), max.

    Neurons are sorted by std (descending) so the "loudest" ones come first. The right column flags neurons that are linearly bright for at least one of the 7 linear features (max |w| / row-max ≥ 0.3).
    """)
    return


@app.cell(hide_code=True)
def _(np, train_acts):
    # Per-neuron summary stats on train_acts — condensed view.
    # Alive neurons (std > 0) shown individually; dead neurons collapsed into one row.
    import pandas as _pd

    _mean = train_acts.mean(0)
    _std  = train_acts.std(0)

    _alive_mask = _std > 0.01
    _alive_idx = np.where(_alive_mask)[0]
    _dead_idx  = np.where(~_alive_mask)[0]

    _alive_rows = _pd.DataFrame({
        "neuron": _alive_idx,
        "mean":   _mean[_alive_idx],
        "std":    _std[_alive_idx],
    }).sort_values("std", ascending=False)

    _dead_row = _pd.DataFrame([{
        "neuron": f"{len(_dead_idx)} dead neurons",
        "mean":   float(_mean[_dead_idx].mean()) if len(_dead_idx) else 0.0,
        "std":    float(_std[_dead_idx].mean())  if len(_dead_idx) else 0.0,
    }])

    neuron_stats = _pd.concat([_alive_rows, _dead_row], ignore_index=True)
    neuron_stats
    return


if __name__ == "__main__":
    app.run()
