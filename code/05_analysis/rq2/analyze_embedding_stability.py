#!/usr/bin/env python3
"""Compute preregistered vision-representation stability analyses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "embedding" / "raw"
OUT = ROOT / "embedding" / "analysis"
SEED = 42
N_BOOT = 1000
N_PERM = 1000


def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def mean_pair_distance(x: np.ndarray) -> float:
    n = len(x)
    return float(1 - (np.square(x.sum(axis=0)).sum() - n) / (n * (n - 1)))


def ci(values: np.ndarray) -> tuple[float, float]:
    return tuple(np.quantile(values, [.025, .975]).tolist())


def load_model(stem: str) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict]:
    manifest = json.loads((RAW / f"{stem}.manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("completed_images") != 9384:
        raise RuntimeError(f"incomplete embedding run: {stem}")
    metadata = pd.read_csv(RAW / f"{stem}.csv")
    archive = np.load(RAW / f"{stem}.npz")
    arrays = {name: normalize(archive[name]) for name in archive.files}
    return metadata, arrays, manifest


def stability(metadata: pd.DataFrame, x: np.ndarray, model: str, representation: str):
    rng = np.random.default_rng(SEED)
    grid_idx = metadata.index[metadata.experiment.eq("grid")].to_numpy()
    grid = metadata.loc[grid_idx].copy()
    conditions = sorted(grid.condition.unique())
    summary_rows, retrieval_rows, detail_rows = [], [], []
    for language in sorted(grid.language.unique()):
        lang = grid[grid.language.eq(language)]
        snippets = sorted(lang.rq0_id.unique())
        index = {(r.rq0_id, r.condition): int(i) for i, r in lang.iterrows()}
        cube = np.stack([[x[index[(snippet, condition)]] for condition in conditions]
                         for snippet in snippets])  # snippet, condition, dim
        n, k, _ = cube.shape
        intra_snippet = np.array([mean_pair_distance(cube[i]) for i in range(n)])
        detail_rows.extend({"model": model, "representation": representation, "language": language,
                            "rq0_id": snippet, "intra_mean_cosine_distance": float(value)}
                           for snippet, value in zip(snippets, intra_snippet))
        inter_condition = np.array([mean_pair_distance(cube[:, j]) for j in range(k)])
        intra, inter = intra_snippet.mean(), inter_condition.mean()

        counts = rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT)
        intra_boot = counts @ intra_snippet / n
        inter_boot_parts = []
        for j in range(k):
            distance = 1 - cube[:, j] @ cube[:, j].T
            np.fill_diagonal(distance, 0)
            inter_boot_parts.append(np.einsum("bi,ij,bj->b", counts, distance, counts) / (n * (n - 1)))
        inter_boot = np.mean(inter_boot_parts, axis=0)
        ratio_boot = intra_boot / inter_boot
        intra_ci, inter_ci, ratio_ci = ci(intra_boot), ci(inter_boot), ci(ratio_boot)

        # Every ordered rendering-condition pair is a retrieval task.
        correct = []
        prediction_sets = []
        for q in range(k):
            for g in range(k):
                if q == g:
                    continue
                nearest = np.argmax(cube[:, q] @ cube[:, g].T, axis=1)
                correct.extend(nearest == np.arange(n))
                prediction_sets.append((q, g, nearest))
        retrieval = float(np.mean(correct))
        correct_matrix = np.stack([nearest == np.arange(n) for _, _, nearest in prediction_sets])
        null = np.empty(N_PERM)
        for p in range(N_PERM):
            labels = [np.arange(n)] + [rng.permutation(n) for _ in range(1, k)]
            hits = 0
            total = 0
            for q, g, nearest in prediction_sets:
                hits += int(np.sum(labels[g][nearest] == labels[q]))
                total += n
            null[p] = hits / total
        retrieval_boot = (counts @ correct_matrix.sum(axis=0)) / (n * len(prediction_sets))
        retrieval_ci = ci(retrieval_boot)
        summary_rows.append({
            "model": model, "representation": representation, "language": language,
            "n_snippets": n, "n_conditions": k, "intra_mean_cosine_distance": intra,
            "intra_boot_ci_low": intra_ci[0], "intra_boot_ci_high": intra_ci[1],
            "inter_mean_cosine_distance": inter, "inter_boot_ci_low": inter_ci[0],
            "inter_boot_ci_high": inter_ci[1], "ratio_intra_inter": intra / inter,
            "ratio_boot_ci_low": ratio_ci[0], "ratio_boot_ci_high": ratio_ci[1],
            "same_snippet_top1": retrieval, "retrieval_ci_low": retrieval_ci[0],
            "retrieval_ci_high": retrieval_ci[1], "retrieval_chance": 1 / n,
            "retrieval_permutation_p": (1 + int(np.sum(null >= retrieval))) / (N_PERM + 1),
            "bootstrap_replicates": N_BOOT, "permutation_replicates": N_PERM,
        })
        for q, condition in enumerate(conditions):
            local = []
            for g in range(k):
                if q != g:
                    nearest = np.argmax(cube[:, q] @ cube[:, g].T, axis=1)
                    local.extend(nearest == np.arange(n))
            retrieval_rows.append({"model": model, "representation": representation,
                                   "language": language, "query_condition": condition,
                                   "n_queries": len(local), "same_snippet_top1": np.mean(local)})
    return summary_rows, retrieval_rows, detail_rows


def perturbation(metadata: pd.DataFrame, x: np.ndarray, model: str, representation: str):
    rng = np.random.default_rng(SEED)
    rows, details = [], []
    baseline_name = "monokai_dark__fs20__wrap80__lnon"
    base_meta = metadata[(metadata.experiment == "grid") & metadata.condition.eq(baseline_name)]
    base_index = {(r.rq0_id, r.language): int(i) for i, r in base_meta.iterrows()}
    pert_meta = metadata[metadata.experiment.eq("perturbation")]
    for (language, condition), part in pert_meta.groupby(["language", "condition"]):
        distances = []
        for i, row in part.iterrows():
            b = base_index[(row.rq0_id, language)]
            distances.append(1 - float(x[b] @ x[int(i)]))
        distances = np.asarray(distances)
        details.extend({"model": model, "representation": representation, "language": language,
                        "condition": condition, "rq0_id": snippet, "baseline_cosine_distance": float(value)}
                       for snippet, value in zip(part.rq0_id, distances))
        boot = rng.choice(distances, (N_BOOT, len(distances)), replace=True).mean(axis=1)
        low, high = ci(boot)
        family = "blur" if condition.startswith("gaussian") else "layout"
        rows.append({"model": model, "representation": representation, "language": language,
                     "condition": condition, "family": family, "n_snippets": len(distances),
                     "mean_baseline_cosine_distance": distances.mean(),
                     "median_baseline_cosine_distance": np.median(distances),
                     "bootstrap_ci_low": low, "bootstrap_ci_high": high})
    return rows, details


def plots(stability_frame: pd.DataFrame, intra: pd.DataFrame, perturb: pd.DataFrame) -> None:
    primary = intra[intra.representation.eq("projected_visual_mean")]
    models = sorted(primary.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(12, 4.5), sharey=True, constrained_layout=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        for language, data in primary[primary.model.eq(model)].groupby("language"):
            ax.hist(data.intra_mean_cosine_distance, bins=30, alpha=.45, density=True, label=language)
        ax.set(title=model.split("/")[-1], xlabel="Intra-snippet cosine distance")
        ax.legend(frameon=False)
    axes[0].set_ylabel("Density")
    fig.savefig(OUT / "d_intra_distance_distributions.png", dpi=200)
    plt.close(fig)

    primary_p = perturb[perturb.representation.eq("projected_visual_mean")]
    order = ["no_indent", "no_blank_lines", "gaussian_sigma_1", "gaussian_sigma_2", "gaussian_sigma_4"]
    fig, axes = plt.subplots(1, len(models), figsize=(13, 4.5), sharey=True, constrained_layout=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        for language, data in primary_p[primary_p.model.eq(model)].groupby("language"):
            data = data.set_index("condition").reindex(order)
            ax.plot(order, data.mean_baseline_cosine_distance, marker="o", label=language)
        ax.set(title=model.split("/")[-1], xlabel="Perturbation")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Mean baseline cosine distance")
    fig.savefig(OUT / "d_blur_layout_distances.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.name.removesuffix(".manifest.json") for p in RAW.glob("*__full_20260722.manifest.json"))
    if len(stems) != 2:
        raise RuntimeError(f"expected two complete full manifests, found {stems}")
    stability_rows, retrieval_rows, intra_rows, perturb_rows, perturb_detail_rows, inventory = [], [], [], [], [], []
    for stem in stems:
        metadata, arrays, manifest = load_model(stem)
        model = manifest["model"]
        for representation, x in arrays.items():
            s, r, detail = stability(metadata, x, model, representation)
            stability_rows.extend(s)
            retrieval_rows.extend(r)
            intra_rows.extend(detail)
            perturb_summary, perturb_detail = perturbation(metadata, x, model, representation)
            perturb_rows.extend(perturb_summary)
            perturb_detail_rows.extend(perturb_detail)
        inventory.append({"model": model, "revision": manifest["model_revision"],
                          "representations": sorted(arrays), "rows": len(metadata)})
    pd.DataFrame(stability_rows).to_csv(OUT / "d_embedding_stability.csv", index=False)
    pd.DataFrame(retrieval_rows).to_csv(OUT / "d_retrieval_by_condition.csv", index=False)
    intra_frame = pd.DataFrame(intra_rows)
    perturb_frame = pd.DataFrame(perturb_rows)
    intra_frame.to_csv(OUT / "d_intra_snippet_distances.csv", index=False)
    perturb_frame.to_csv(OUT / "d_perturbation_distance.csv", index=False)
    pd.DataFrame(perturb_detail_rows).to_csv(OUT / "d_perturbation_snippet_distances.csv", index=False)
    plots(pd.DataFrame(stability_rows), intra_frame, perturb_frame)
    (OUT / "D_INTEGRITY.json").write_text(json.dumps({"models": inventory,
        "bootstrap_replicates": N_BOOT, "permutation_replicates": N_PERM,
        "fixed_ratio_thresholds_used": False}, indent=2) + "\n")
    print(pd.DataFrame(stability_rows).to_string(index=False))


if __name__ == "__main__":
    main()
