# BlueDot Technical AI Safety Puzzle #1

Our submission for the [BlueDot Technical AI Safety Puzzle](https://bluedot.org/puzzles/technical-ai-safety).

The full writeup is in **[Report.md](Report.md)**.

## Notebooks

All experiments are [marimo](https://marimo.io) notebooks.

- **[puzzle.py](puzzle.py)** — Q1 + Q2: identify the non-linear feature (`country`) and characterize its quadric representation.
- **[q3_superposition.py](q3_superposition.py)** — Q3: sweep the hidden-layer-2 width `k` and observe head-only cases.
- **[q3_concentric_shells.py](q3_concentric_shells.py)** — Q3: train an encoder that places classes on nested shells using `L_radial` + `L_centroid_gap`.
- **[q3_grl_adversarial.py](q3_grl_adversarial.py)** — Q3: gradient-reversal attempt (failed); see [q3-grl-adversarial.md](q3-grl-adversarial.md) and [adversarial-grl-math.md](adversarial-grl-math.md).
- **[q3_two_moons.py](q3_two_moons.py)** — sanity check on synthetic two-moons.

## Run

```bash
pip install marimo sentence-transformers torch scikit-learn matplotlib
marimo edit puzzle.py        # or any of the q3_*.py notebooks
```

## Repo contents

- `model.pt` — trained classifier state dict (~150 KB).
- `data/train.jsonl`, `data/test.jsonl` — 7000 + 1500 lines of `{"text": ..., "labels": [...]}`.
- `feature_names.json` — the eight feature names, indexed 0–7.
- `images/` — figures referenced from `Report.md`.
