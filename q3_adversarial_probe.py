# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "torch",
#     "sentence-transformers",
#     "scikit-learn",
#     "matplotlib",
#     "numpy",
# ]
# ///

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
    # Q3 — Adversarial probe-resistance training
    "
          "Retrain the MLP head with an adversarial probe at hidden 2 so that "
          "linear / degree-2 polynomial probes fail on feature F, while the "
          "model's own head still predicts F.
    """)
    return


@app.cell
def _(nn):
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(384, 64), nn.ReLU(),
                nn.Linear(64, 64),  nn.ReLU(),
                nn.Linear(64, 64),  nn.ReLU(),
                nn.Linear(64, 64),  nn.ReLU(),
                nn.Linear(64, 8),
            )

        def hidden2(self, x):
            return self.layers[:6](x)

        def forward(self, x):
            return self.layers(x)

    return (Head,)


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
def _(feature_names, mo):
    feature_choice = mo.ui.dropdown(
        options=feature_names,
        value="sentiment",
        label="Target feature F",
    )
    feature_choice
    return (feature_choice,)


@app.cell
def _(mo):
    adv_choice = mo.ui.dropdown(
        options=["linear", "degree-2 polynomial"],
        value="linear",
        label="Adversary probe form",
    )
    adv_choice
    return (adv_choice,)


@app.cell
def _(mo):
    lambda_max_slider = mo.ui.slider(
        start=0.0, stop=5.0, step=0.1, value=1.0,
        label="lambda_max (adversary strength after ramp)",
        show_value=True,
    )
    lambda_max_slider
    return (lambda_max_slider,)


@app.cell
def _(torch):
    # Gradient Reversal Layer: identity in forward, multiplies gradient by -lam in backward.
    class _GRL(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lam):
            ctx.lam = lam
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad):
            return -ctx.lam * grad, None

    def grl(x, lam):
        return _GRL.apply(x, lam)

    return


@app.cell
def _(torch):
    def poly2_features(x):
        # x: (B, n) -> (B, n + n*(n+1)/2) with linear + all unique pairwise products (incl. squares).
        n = x.shape[1]
        outer = x.unsqueeze(2) * x.unsqueeze(1)
        iu = torch.triu_indices(n, n)
        sq = outer[:, iu[0], iu[1]]
        return torch.cat([x, sq], dim=1)

    return (poly2_features,)


@app.cell
def _(nn, poly2_features):
    class Adversary(nn.Module):
        def __init__(self, kind, hidden_dim=64):
            super().__init__()
            self.kind = kind
            in_dim = hidden_dim if kind == "linear" else hidden_dim + hidden_dim * (hidden_dim + 1) // 2
            self.linear = nn.Linear(in_dim, 1)

        def forward(self, h):
            z = h if self.kind == "linear" else poly2_features(h)
            return self.linear(z).squeeze(-1)

    return (Adversary,)


@app.cell
def _(
    Adversary,
    Head,
    adv_choice,
    feature_choice,
    feature_names,
    lambda_max_slider,
    nn,
    te_emb,
    te_labels,
    torch,
    tr_emb,
    tr_labels,
):
    torch.manual_seed(0)

    f_idx = feature_names.index(feature_choice.value)
    model = Head()
    adversary = Adversary(adv_choice.value)

    opt_body = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt_adv  = torch.optim.Adam(adversary.parameters(), lr=5e-3)
    bce = nn.BCEWithLogitsLoss()

    tr_X = tr_emb
    te_X = te_emb
    tr_y_all = torch.from_numpy(tr_labels)
    te_y_all = torch.from_numpy(te_labels)
    tr_y_F = tr_y_all[:, f_idx]

    n_epochs = 50
    batch_size = 128
    K_inner = 10
    reinit_every = 5      # epochs between full adversary reinits
    n_train = tr_X.shape[0]
    lam_max = lambda_max_slider.value

    losses_main = []
    losses_adv = []

    for epoch in range(n_epochs):
        lam = lam_max * (epoch + 1) / n_epochs

        # Periodic adversary reinit: force body to defend against many linear classifiers, not one.
        if epoch > 0 and epoch % reinit_every == 0:
            adversary = Adversary(adv_choice.value)
            opt_adv = torch.optim.Adam(adversary.parameters(), lr=5e-3)

        perm = torch.randperm(n_train)
        ep_main = 0.0
        ep_adv = 0.0
        n_batches = 0

        model.train()
        adversary.train()
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            x = tr_X[idx]
            y_all = tr_y_all[idx]
            y_F = tr_y_F[idx]

            with torch.no_grad():
                h2_det = model.hidden2(x)
            for _ in range(K_inner):
                adv_logits = adversary(h2_det)
                l_adv_inner = bce(adv_logits, y_F)
                opt_adv.zero_grad()
                l_adv_inner.backward()
                opt_adv.step()

            h2_live = model.hidden2(x)
            logits = model.layers[6:](h2_live)
            loss_main = bce(logits, y_all)
            adv_logits_live = adversary(h2_live)
            loss_adv_live = bce(adv_logits_live, y_F)
            loss_body = loss_main - lam * loss_adv_live

            opt_body.zero_grad()
            loss_body.backward()
            opt_body.step()

            ep_main += loss_main.item()
            ep_adv += loss_adv_live.item()
            n_batches += 1

        losses_main.append(ep_main / n_batches)
        losses_adv.append(ep_adv / n_batches)

    print(f"Done. F={feature_choice.value}  adversary={adv_choice.value}  lambda_max={lam_max}  K_inner={K_inner}  reinit_every={reinit_every}")
    print(f"Final main BCE = {losses_main[-1]:.3f}   adversary BCE = {losses_adv[-1]:.3f}  (chance \u2248 0.693)")

    return (
        f_idx,
        losses_adv,
        losses_main,
        model,
        n_epochs,
        te_X,
        te_y_all,
        tr_X,
        tr_y_all,
    )


@app.cell
def _(losses_adv, losses_main, n_epochs, np):
    import matplotlib.pyplot as plt

    fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
    ax_loss.plot(range(1, n_epochs + 1), losses_main, label="main BCE (8-feature)")
    ax_loss.plot(range(1, n_epochs + 1), losses_adv, label="adversary BCE (on F)")
    ax_loss.axhline(np.log(2), linestyle="--", color="gray", linewidth=1, label="chance (ln 2)")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Training losses (adversary should rise toward chance)")
    ax_loss.legend()
    fig_loss.tight_layout()
    fig_loss
    return


app._unparsable_cell(
    r"""
     r
    """,
    name="_"
)


@app.cell
def _(
    Head,
    adv_choice,
    f_idx,
    feature_choice,
    lambda_max_slider,
    model,
    np,
    te_X,
    te_y_all,
    torch,
    tr_X,
    tr_y_all,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    model.eval()
    with torch.no_grad():
        tr_h2_new = model.hidden2(tr_X).numpy()
        te_h2_new = model.hidden2(te_X).numpy()
        te_logits_new = model(te_X).numpy()

    original = Head()
    original.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=False))
    original.eval()
    with torch.no_grad():
        tr_h2_orig = original.layers[:6](tr_X).numpy()
        te_h2_orig = original.layers[:6](te_X).numpy()
        te_logits_orig = original(te_X).numpy()

    y_tr_all = tr_y_all.numpy().astype(int)
    y_te_all = te_y_all.numpy().astype(int)
    y_tr_F = y_tr_all[:, f_idx]
    y_te_F = y_te_all[:, f_idx]

    def _lin():
        return LogisticRegression(max_iter=1000)

    def _poly():
        return make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                             LogisticRegression(max_iter=2000, C=0.1))

    def _probe(maker, Xtr, ytr, Xte, yte):
        c = maker()
        c.fit(Xtr, ytr)
        return c.score(Xte, yte)

    own_F_orig = float(((te_logits_orig[:, f_idx] > 0).astype(int) == y_te_F).mean())
    own_F_new = float(((te_logits_new[:, f_idx] > 0).astype(int) == y_te_F).mean())

    others = [i for i in range(8) if i != f_idx]
    avg_others_orig = float(np.mean([
        _probe(_lin, tr_h2_orig, y_tr_all[:, i], te_h2_orig, y_te_all[:, i])
        for i in others
    ]))
    avg_others_new = float(np.mean([
        _probe(_lin, tr_h2_new, y_tr_all[:, i], te_h2_new, y_te_all[:, i])
        for i in others
    ]))

    rows = [
        ("Linear LogReg on F", _probe(_lin, tr_h2_orig, y_tr_F, te_h2_orig, y_te_F),
                               _probe(_lin, tr_h2_new,  y_tr_F, te_h2_new,  y_te_F)),
        ("Degree-2 poly LogReg on F", _probe(_poly, tr_h2_orig, y_tr_F, te_h2_orig, y_te_F),
                                       _probe(_poly, tr_h2_new,  y_tr_F, te_h2_new,  y_te_F)),
        ("Model\u2019s own head on F", own_F_orig, own_F_new),
        ("Avg linear probe on other 7", avg_others_orig, avg_others_new),
    ]

    print(f"\nF = {feature_choice.value}   adversary = {adv_choice.value}   lambda_max = {lambda_max_slider.value}")
    print(f"  {'probe':<32} {'original':>10} {'adv-trained':>14}")
    for name, a, b in rows:
        print(f"  {name:<32} {a:>10.3f} {b:>14.3f}")

    return


if __name__ == "__main__":
    app.run()
