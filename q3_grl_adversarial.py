# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "torch",
#     "numpy",
#     "scikit-learn",
#     "matplotlib",
#     "pandas",
#     "sentence-transformers",
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
    import torch.nn.functional as Fnn
    from torch.autograd import Function
    from sentence_transformers import SentenceTransformer

    return Fnn, Function, SentenceTransformer, json, mo, nn, np, torch


@app.cell
def _(mo):
    mo.md("""
    # Q3 — Adversarial probe-resistance via GRL

    Train an encoder so that a linear probe fails on F=`sentiment`,
    while the model's own head still predicts F well.

    **The failure mode this notebook is built to surface.** A single
    slowly-updated in-loop probe will appear defeated even when the
    encoder has *not* removed F — a freshly retrained held-out probe
    then recovers F. We track both, side by side.

    **Defenses (all toggleable):**

    - **k_inner**: probe gradient steps per encoder step (probe stays
      near-optimal so the encoder must actually scrub F, not just fool
      a lagging adversary).
    - **reinit_every**: re-initialise every probe every N epochs to
      break out of stale local minima.
    - **K_probes**: ensemble of K probes; the encoder is pushed against
      the *strongest* one each step.
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
    lambda_slider = mo.ui.slider(
        start=0.0, stop=2.0, step=0.05, value=0.5,
        label="λ — GRL weight", show_value=True,
    )
    k_inner_slider = mo.ui.slider(
        start=1, stop=20, step=1, value=5,
        label="k_inner — probe steps per encoder step", show_value=True,
    )
    reinit_slider = mo.ui.slider(
        start=0, stop=20, step=1, value=5,
        label="reinit_every (epochs; 0 = never)", show_value=True,
    )
    K_probes_slider = mo.ui.slider(
        start=1, stop=8, step=1, value=3,
        label="K probes (ensemble size)", show_value=True,
    )
    n_epochs_slider = mo.ui.slider(
        start=5, stop=80, step=5, value=30,
        label="epochs", show_value=True,
    )
    mo.vstack([lambda_slider, k_inner_slider, reinit_slider, K_probes_slider, n_epochs_slider])
    return (
        K_probes_slider,
        k_inner_slider,
        lambda_slider,
        n_epochs_slider,
        reinit_slider,
    )


@app.cell
def _(mo):
    mo.md("""
    **Naive GRL** = `k_inner=1, reinit_every=0, K_probes=1`. This is the
    config the Elazar & Goldberg (2018) result shows gives a false win;
    useful as a baseline to compare against.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Architecture
    """)
    return


@app.cell
def _(Fnn, Function, nn, torch):
    class GradReverse(Function):
        @staticmethod
        def forward(ctx, x, lambda_):
            ctx.lambda_ = float(lambda_)
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.lambda_ * grad_output, None

    def grad_reverse(x, lambda_=1.0):
        return GradReverse.apply(x, lambda_)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(384, 64)
            self.l2 = nn.Linear(64, 64)
            self.l3 = nn.Linear(64, 64)   # hidden 2 — kept signed (no ReLU)

        def forward(self, x):
            h = Fnn.relu(self.l1(x))
            h = Fnn.relu(self.l2(h))
            return self.l3(h)

    class MainHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.l4 = nn.Linear(64, 64)
            self.l5 = nn.Linear(64, 8)

        def forward(self, h):
            return self.l5(Fnn.relu(self.l4(h)))

    class LinearProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(64, 1)

        def forward(self, h):
            return self.w(h).squeeze(-1)

    def fresh_probes(K, seed_base=0):
        probes = []
        for k in range(K):
            torch.manual_seed(seed_base + k * 17 + 1)
            probes.append(LinearProbe())
        return probes

    return Encoder, MainHead, fresh_probes, grad_reverse


@app.cell
def _(mo):
    mo.md("""
    ## Data & embeddings
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    F_choice = mo.ui.dropdown(
        options=["number","question","color","food","sentiment","country","person","body_part"],
        value="person",
        label="Target feature F",
    )
    F_choice
    return (F_choice,)


@app.cell
def _(F_choice, SentenceTransformer, json, np, torch):
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

    F_NAME = F_choice.value
    f_idx_g = feature_names.index(F_NAME)
    print(f"train: {tuple(tr_emb.shape)}   test: {tuple(te_emb.shape)}   target F = '{F_NAME}' (idx {f_idx_g})")
    return F_NAME, f_idx_g, feature_names, te_emb, te_labels, tr_emb, tr_labels


@app.cell
def _(mo):
    mo.md("""
    ## Training

    Per batch:

    1. **Probe update.** Detach `h` so probe gradients don't reach the
       encoder. Run `k_inner` steps for *each* probe in the ensemble.
    2. **Encoder + head update.** Forward through GRL on hidden 2;
       the encoder receives a *reversed* gradient from the probe that
       is currently strongest (lowest loss). Main BCE on all 8 features
       is added unreversed.
    3. **End of epoch.** Refit a *fresh* `LogisticRegression` on frozen
       hidden 2 to get the held-out probe accuracy — that's the real
       "did we scrub F" metric.
    """)
    return


@app.cell
def _(
    Encoder,
    F_NAME,
    K_probes_slider,
    MainHead,
    f_idx_g,
    fresh_probes,
    grad_reverse,
    k_inner_slider,
    lambda_slider,
    n_epochs_slider,
    nn,
    np,
    reinit_slider,
    te_emb,
    te_labels,
    torch,
    tr_emb,
    tr_labels,
):
    from sklearn.linear_model import LogisticRegression as _LR_g

    LAMBDA = float(lambda_slider.value)
    K_INNER = int(k_inner_slider.value)
    REINIT = int(reinit_slider.value)
    K_PROBES = int(K_probes_slider.value)
    N_EPOCHS = int(n_epochs_slider.value)
    BS = 128

    torch.manual_seed(0)
    encoder = Encoder()
    main_head = MainHead()
    probes = fresh_probes(K_PROBES, seed_base=0)

    opt_main = torch.optim.Adam(
        list(encoder.parameters()) + list(main_head.parameters()), lr=1e-3
    )
    opt_probes = [torch.optim.Adam(p.parameters(), lr=1e-3) for p in probes]

    bce_multi = nn.BCEWithLogitsLoss()
    bce_bin = nn.BCEWithLogitsLoss()

    tr_y = torch.from_numpy(tr_labels)
    tr_y_F = tr_y[:, f_idx_g]
    n_train = tr_emb.shape[0]

    hist_main = []
    hist_probe_inloop = []
    hist_probe_heldout = []
    hist_mu_gap = []

    for epoch in range(N_EPOCHS):
        if REINIT > 0 and epoch > 0 and epoch % REINIT == 0:
            probes = fresh_probes(K_PROBES, seed_base=epoch * 31)
            opt_probes = [torch.optim.Adam(p.parameters(), lr=1e-3) for p in probes]

        perm = torch.randperm(n_train)
        ep_main = 0.0
        ep_batches = 0

        for start in range(0, n_train, BS):
            idx = perm[start:start + BS]
            x = tr_emb[idx]
            y_all = tr_y[idx]
            y_F = tr_y_F[idx]

            # ---- 1. probe inner loop ----
            with torch.no_grad():
                h_det = encoder(x)
            h_det = h_det.detach()
            for _ in range(K_INNER):
                for p_idx, probe in enumerate(probes):
                    opt_probes[p_idx].zero_grad()
                    bce_bin(probe(h_det), y_F).backward()
                    opt_probes[p_idx].step()

            # ---- 2. encoder + head step ----
            h_live = encoder(x)
            main_logits = main_head(h_live)
            L_main = bce_multi(main_logits, y_all)

            with torch.no_grad():
                losses_for_pick = [bce_bin(p(h_live), y_F).item() for p in probes]
            strongest = int(np.argmin(losses_for_pick))
            h_rev = grad_reverse(h_live, LAMBDA)
            L_adv = bce_bin(probes[strongest](h_rev), y_F)

            opt_main.zero_grad()
            (L_main + L_adv).backward()
            opt_main.step()

            ep_main += L_main.item()
            ep_batches += 1

        # ---- 3. end-of-epoch diagnostics ----
        with torch.no_grad():
            tr_h = encoder(tr_emb).numpy()
            te_h = encoder(te_emb).numpy()

        ho = _LR_g(max_iter=1000).fit(tr_h, tr_labels.astype(int)[:, f_idx_g])
        heldout_acc = float(ho.score(te_h, te_labels.astype(int)[:, f_idx_g]))

        with torch.no_grad():
            te_h_t = encoder(te_emb)
            inloop_accs = []
            for p in probes:
                pred = (p(te_h_t).numpy() > 0).astype(int)
                inloop_accs.append(float((pred == te_labels.astype(int)[:, f_idx_g]).mean()))
        inloop_acc = max(inloop_accs)

        c = tr_h - tr_h.mean(0, keepdims=True)
        nrm = np.linalg.norm(c, axis=1, keepdims=True) + 1e-9
        dirs_full = c / nrm
        yF_tr_int = tr_labels.astype(int)[:, f_idx_g]
        mu_p = dirs_full[yF_tr_int == 1].mean(0)
        mu_n = dirs_full[yF_tr_int == 0].mean(0)
        mu_gap = float(np.linalg.norm(mu_p - mu_n))

        hist_main.append(ep_main / ep_batches)
        hist_probe_inloop.append(inloop_acc)
        hist_probe_heldout.append(heldout_acc)
        hist_mu_gap.append(mu_gap)

        if (epoch + 1) % max(1, N_EPOCHS // 10) == 0 or epoch == 0:
            print(f"epoch {epoch+1:>3}/{N_EPOCHS}  "
                  f"L_main={hist_main[-1]:.3f}  "
                  f"inloop={inloop_acc:.3f}  heldout={heldout_acc:.3f}  "
                  f"‖μ₁−μ₀‖={mu_gap:.3f}")

    print(f"\nDone. λ={LAMBDA}  k_inner={K_INNER}  reinit_every={REINIT}  K={K_PROBES}  F={F_NAME}")
    return (
        encoder,
        hist_main,
        hist_mu_gap,
        hist_probe_heldout,
        hist_probe_inloop,
        main_head,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Diagnostics

    The two curves to watch:

    - **In-loop probe** — the probe in the training loop. Falls quickly
      when the encoder finds *any* way to evade this specific probe.
    - **Held-out probe** — a fresh `LogisticRegression` retrained on the
      current encoder's hidden 2 at the end of every epoch. **This is
      the real metric.** If it stays high while the in-loop probe
      drops, the encoder is fooling the probe rather than removing F.
    """)
    return


@app.cell
def _(hist_main, hist_mu_gap, hist_probe_heldout, hist_probe_inloop):
    import matplotlib.pyplot as plt

    fig_diag, axes_diag = plt.subplots(1, 3, figsize=(15, 4))

    axes_diag[0].plot(hist_main, label="L_main (BCE)", color="tab:gray")
    axes_diag[0].set_xlabel("epoch")
    axes_diag[0].set_ylabel("loss")
    axes_diag[0].set_yscale("log")
    axes_diag[0].set_title("Main task loss")
    axes_diag[0].legend()

    axes_diag[1].plot(hist_probe_inloop, label="in-loop probe", color="tab:orange")
    axes_diag[1].plot(hist_probe_heldout, label="held-out probe (real metric)", color="tab:blue")
    axes_diag[1].axhline(0.5, color="k", ls="--", lw=0.6, alpha=0.6)
    axes_diag[1].set_xlabel("epoch")
    axes_diag[1].set_ylabel("acc on F=sentiment")
    axes_diag[1].set_ylim(0.4, 1.02)
    axes_diag[1].set_title("Probe accuracy — in-loop vs held-out")
    axes_diag[1].legend()

    axes_diag[2].plot(hist_mu_gap, color="tab:green")
    axes_diag[2].set_xlabel("epoch")
    axes_diag[2].set_ylabel("‖μ₁ − μ₀‖")
    axes_diag[2].set_title("Angular class-mean gap on hidden 2")

    fig_diag.tight_layout()
    fig_diag
    return (plt,)


@app.cell
def _(mo):
    mo.md("""
    ## Per-feature: linear probe vs model's own head
    """)
    return


@app.cell
def _(
    F_NAME,
    encoder,
    feature_names,
    main_head,
    np,
    plt,
    te_emb,
    te_labels,
    torch,
    tr_emb,
    tr_labels,
):
    from sklearn.linear_model import LogisticRegression as _LR_pf
    import matplotlib.patches as _mpatches

    encoder.eval(); main_head.eval()
    with torch.no_grad():
        tr_h_pf = encoder(tr_emb).numpy()
        te_h_pf = encoder(te_emb).numpy()
        te_logits_pf = main_head(torch.from_numpy(te_h_pf)).numpy()

    y_tr_int = tr_labels.astype(int)
    y_te_int = te_labels.astype(int)

    probe_accs = []
    head_accs = []
    for fi in range(8):
        lr = _LR_pf(max_iter=2000).fit(tr_h_pf, y_tr_int[:, fi])
        probe_accs.append(float(lr.score(te_h_pf, y_te_int[:, fi])))
        head_accs.append(float(((te_logits_pf[:, fi] > 0).astype(int) == y_te_int[:, fi]).mean()))

    xb = np.arange(len(feature_names))
    w = 0.4
    _cmap = plt.get_cmap("tab10")
    _colors = [_cmap(i) for i in range(len(feature_names))]

    fig_bars, ax_bars = plt.subplots(figsize=(10, 4.5))
    for _i, (_name, _pacc, _oacc, _color) in enumerate(zip(feature_names, probe_accs, head_accs, _colors)):
        ax_bars.bar(xb[_i] - w / 2, _pacc, w, color=_color, edgecolor="black", linewidth=0.5)
        ax_bars.bar(xb[_i] + w / 2, _oacc, w, color=_color, edgecolor="black", linewidth=0.5, hatch="///")

    ax_bars.set_xticks(xb)
    ax_bars.set_xticklabels(feature_names, rotation=30, ha="right")
    ax_bars.set_ylim(0, 1.05)
    ax_bars.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax_bars.set_ylabel("test accuracy")
    ax_bars.set_title(f"Linear probe (solid) vs model\u2019s own head (hatched) per feature — F = \'{F_NAME}\'")
    _legend = [
        _mpatches.Patch(facecolor="lightgray", edgecolor="black", label="probe"),
        _mpatches.Patch(facecolor="lightgray", edgecolor="black", hatch="///", label="own head"),
    ]
    ax_bars.legend(handles=_legend, loc="lower right")

    _f_idx_bars = feature_names.index(F_NAME)
    ax_bars.annotate("target F\n(head < probe)",
                     xy=(_f_idx_bars, max(probe_accs[_f_idx_bars], head_accs[_f_idx_bars])),
                     xytext=(_f_idx_bars, 0.30),
                     ha="center", fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    fig_bars.tight_layout()
    fig_bars
    return


@app.cell
def _(mo):
    mo.md("""
    ## Ablation sweep — does defending against staleness matter?

    Holds λ fixed. Four configs:

    | config            | k_inner | reinit_every | K_probes |
    |-------------------|--------:|-------------:|---------:|
    | naive GRL         | 1       | 0            | 1        |
    | + k_inner         | 5       | 0            | 1        |
    | + reinit          | 5       | 5            | 1        |
    | + ensemble (all)  | 5       | 5            | 3        |

    Each row reports the in-loop probe acc *and* the held-out probe
    acc. The expectation: in-loop accuracy drops to ~chance in every
    row (the encoder always learns to fool *the probe it sees*); the
    held-out accuracy reveals which configs actually scrubbed F.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    run_ablation_btn = mo.ui.run_button(label="Run defense ablation")
    run_ablation_btn
    return (run_ablation_btn,)


@app.cell
def _(
    Encoder,
    MainHead,
    f_idx_g,
    fresh_probes,
    grad_reverse,
    lambda_slider,
    mo,
    n_epochs_slider,
    nn,
    np,
    run_ablation_btn,
    te_emb,
    te_labels,
    torch,
    tr_emb,
    tr_labels,
):
    mo.stop(not run_ablation_btn.value, mo.md("Press **Run defense ablation** to start."))

    from sklearn.linear_model import LogisticRegression as _LR_ab
    import pandas as _pd_ab

    LAMBDA_AB = float(lambda_slider.value)
    N_EPOCHS_AB = int(n_epochs_slider.value)
    BS_AB = 128

    CONFIGS = [
        {"name": "naive GRL",         "k_inner": 1, "reinit": 0, "K": 1},
        {"name": "+ k_inner=5",       "k_inner": 5, "reinit": 0, "K": 1},
        {"name": "+ reinit every 5",  "k_inner": 5, "reinit": 5, "K": 1},
        {"name": "+ ensemble (K=3)",  "k_inner": 5, "reinit": 5, "K": 3},
    ]

    def _train_cfg(cfg):
        torch.manual_seed(0)
        enc_ = Encoder()
        head_ = MainHead()
        probes_ = fresh_probes(cfg["K"], seed_base=0)
        opt_main_ = torch.optim.Adam(list(enc_.parameters()) + list(head_.parameters()), lr=1e-3)
        opt_probes_ = [torch.optim.Adam(p.parameters(), lr=1e-3) for p in probes_]
        bce_m = nn.BCEWithLogitsLoss(); bce_b = nn.BCEWithLogitsLoss()
        tr_y_ = torch.from_numpy(tr_labels); tr_y_F_ = tr_y_[:, f_idx_g]
        n_ = tr_emb.shape[0]
        for epoch in range(N_EPOCHS_AB):
            if cfg["reinit"] > 0 and epoch > 0 and epoch % cfg["reinit"] == 0:
                probes_ = fresh_probes(cfg["K"], seed_base=epoch * 31)
                opt_probes_ = [torch.optim.Adam(p.parameters(), lr=1e-3) for p in probes_]
            perm = torch.randperm(n_)
            for s in range(0, n_, BS_AB):
                idx = perm[s:s+BS_AB]
                x_ = tr_emb[idx]; y_all_ = tr_y_[idx]; y_F_ = tr_y_F_[idx]
                with torch.no_grad():
                    h_d = enc_(x_)
                h_d = h_d.detach()
                for _ in range(cfg["k_inner"]):
                    for pi, p_ in enumerate(probes_):
                        opt_probes_[pi].zero_grad()
                        bce_b(p_(h_d), y_F_).backward()
                        opt_probes_[pi].step()
                h_l = enc_(x_)
                L_m_ = bce_m(head_(h_l), y_all_)
                with torch.no_grad():
                    losses = [bce_b(p_(h_l), y_F_).item() for p_ in probes_]
                strong = int(np.argmin(losses))
                h_r = grad_reverse(h_l, LAMBDA_AB)
                L_a_ = bce_b(probes_[strong](h_r), y_F_)
                opt_main_.zero_grad(); (L_m_ + L_a_).backward(); opt_main_.step()
        # evaluation
        enc_.eval(); head_.eval()
        with torch.no_grad():
            tr_h_ = enc_(tr_emb).numpy(); te_h_ = enc_(te_emb).numpy()
            te_lg = head_(torch.from_numpy(te_h_)).numpy()
            # in-loop probe acc (current strongest)
            inloop_ = max(
                float(((p_(torch.from_numpy(te_h_)).numpy() > 0).astype(int)
                        == te_labels.astype(int)[:, f_idx_g]).mean())
                for p_ in probes_
            )
        ho_ = _LR_ab(max_iter=1000).fit(tr_h_, tr_labels.astype(int)[:, f_idx_g])
        heldout_ = float(ho_.score(te_h_, te_labels.astype(int)[:, f_idx_g]))
        head_F_ = float(((te_lg[:, f_idx_g] > 0).astype(int) == te_labels.astype(int)[:, f_idx_g]).mean())
        worst_other_ = min(
            float(((te_lg[:, fi] > 0).astype(int) == te_labels.astype(int)[:, fi]).mean())
            for fi in range(8) if fi != f_idx_g
        )
        # mu gap
        c_ = tr_h_ - tr_h_.mean(0, keepdims=True)
        n_ = np.linalg.norm(c_, axis=1, keepdims=True) + 1e-9
        d_ = c_ / n_
        yF_ = tr_labels.astype(int)[:, f_idx_g]
        mu_gap_ = float(np.linalg.norm(d_[yF_ == 1].mean(0) - d_[yF_ == 0].mean(0)))
        return {
            "in-loop probe": round(inloop_, 3),
            "held-out probe": round(heldout_, 3),
            "head on F": round(head_F_, 3),
            "worst other head": round(worst_other_, 3),
            "‖μ₁−μ₀‖": round(mu_gap_, 3),
        }

    _rows = []
    for cfg in CONFIGS:
        print(f"training: {cfg['name']:<22} k_inner={cfg['k_inner']} reinit={cfg['reinit']} K={cfg['K']}", flush=True)
        r = _train_cfg(cfg)
        _rows.append({"config": cfg["name"], **r})

    ablation_df = _pd_ab.DataFrame(_rows)
    print(ablation_df.to_string(index=False))
    ablation_df
    return


@app.cell
def _(mo):
    mo.md("""
    ## λ-sweep — how strong should the adversarial pressure be?

    Holds the defense stack at (k_inner=5, reinit_every=5, K=3) — i.e.
    the "all defenses on" config. Sweeps λ ∈ {0.0, 0.05, 0.5, 2.0}.
    Reports held-out probe acc, head acc on F, worst other head,
    and ‖μ₁−μ₀‖.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    run_lambda_btn = mo.ui.run_button(label="Run λ-sweep")
    run_lambda_btn
    return (run_lambda_btn,)


@app.cell
def _(
    Encoder,
    MainHead,
    f_idx_g,
    fresh_probes,
    grad_reverse,
    mo,
    n_epochs_slider,
    nn,
    np,
    run_lambda_btn,
    te_emb,
    te_labels,
    torch,
    tr_emb,
    tr_labels,
):
    mo.stop(not run_lambda_btn.value, mo.md("Press **Run λ-sweep** to start."))

    from sklearn.linear_model import LogisticRegression as _LR_l
    import pandas as _pd_l

    LAMBDAS = [0.0, 0.05, 0.5, 2.0]
    K_INNER_L = 5; REINIT_L = 5; K_PROBES_L = 3
    N_EPOCHS_L = int(n_epochs_slider.value)
    BS_L = 128

    def _train_lam(lam):
        torch.manual_seed(0)
        enc_ = Encoder(); head_ = MainHead()
        probes_ = fresh_probes(K_PROBES_L, seed_base=0)
        opt_main_ = torch.optim.Adam(list(enc_.parameters()) + list(head_.parameters()), lr=1e-3)
        opt_probes_ = [torch.optim.Adam(p.parameters(), lr=1e-3) for p in probes_]
        bce_m = nn.BCEWithLogitsLoss(); bce_b = nn.BCEWithLogitsLoss()
        tr_y_ = torch.from_numpy(tr_labels); tr_y_F_ = tr_y_[:, f_idx_g]
        n_ = tr_emb.shape[0]
        for epoch in range(N_EPOCHS_L):
            if REINIT_L > 0 and epoch > 0 and epoch % REINIT_L == 0:
                probes_ = fresh_probes(K_PROBES_L, seed_base=epoch * 31)
                opt_probes_ = [torch.optim.Adam(p.parameters(), lr=1e-3) for p in probes_]
            perm = torch.randperm(n_)
            for s in range(0, n_, BS_L):
                idx = perm[s:s+BS_L]
                x_ = tr_emb[idx]; y_all_ = tr_y_[idx]; y_F_ = tr_y_F_[idx]
                with torch.no_grad():
                    h_d = enc_(x_)
                h_d = h_d.detach()
                for _ in range(K_INNER_L):
                    for pi, p_ in enumerate(probes_):
                        opt_probes_[pi].zero_grad()
                        bce_b(p_(h_d), y_F_).backward()
                        opt_probes_[pi].step()
                h_l = enc_(x_)
                L_m_ = bce_m(head_(h_l), y_all_)
                with torch.no_grad():
                    losses = [bce_b(p_(h_l), y_F_).item() for p_ in probes_]
                strong = int(np.argmin(losses))
                h_r = grad_reverse(h_l, lam)
                L_a_ = bce_b(probes_[strong](h_r), y_F_)
                opt_main_.zero_grad(); (L_m_ + L_a_).backward(); opt_main_.step()
        enc_.eval(); head_.eval()
        with torch.no_grad():
            tr_h_ = enc_(tr_emb).numpy(); te_h_ = enc_(te_emb).numpy()
            te_lg = head_(torch.from_numpy(te_h_)).numpy()
        ho_ = _LR_l(max_iter=1000).fit(tr_h_, tr_labels.astype(int)[:, f_idx_g])
        heldout_ = float(ho_.score(te_h_, te_labels.astype(int)[:, f_idx_g]))
        head_F_ = float(((te_lg[:, f_idx_g] > 0).astype(int) == te_labels.astype(int)[:, f_idx_g]).mean())
        worst_other_ = min(
            float(((te_lg[:, fi] > 0).astype(int) == te_labels.astype(int)[:, fi]).mean())
            for fi in range(8) if fi != f_idx_g
        )
        c_ = tr_h_ - tr_h_.mean(0, keepdims=True)
        nrm_ = np.linalg.norm(c_, axis=1, keepdims=True) + 1e-9
        d_ = c_ / nrm_
        yF_ = tr_labels.astype(int)[:, f_idx_g]
        mu_gap_ = float(np.linalg.norm(d_[yF_ == 1].mean(0) - d_[yF_ == 0].mean(0)))
        return {
            "held-out probe": round(heldout_, 3),
            "head on F": round(head_F_, 3),
            "worst other head": round(worst_other_, 3),
            "‖μ₁−μ₀‖": round(mu_gap_, 3),
        }

    _rows_l = []
    for lam in LAMBDAS:
        print(f"λ={lam:.2f} training...", flush=True)
        _rows_l.append({"λ": lam, **_train_lam(lam)})

    lambda_sweep_df = _pd_l.DataFrame(_rows_l)
    print(lambda_sweep_df.to_string(index=False))
    lambda_sweep_df
    return


if __name__ == "__main__":
    app.run()
