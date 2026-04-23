"""
Plot per-image AKD distributions (histogram + CDF) for mugs, shoes, tshirts.
Reads the CSVs produced by compute_l2_per_keypoint.py.

Usage:
    conda run -n dsd python dsd/results/plot_l2_distributions_from_csv.py
"""

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

CSV_DIR = Path(__file__).parent
CATEGORIES = ["Mugs", "Shoes", "Tshirts"]
XLIMS = {"Mugs": 50, "Shoes": 50, "Tshirts": 105}
LABEL_MAP = {
    "REAL": "real\ndata",
    "RANDOM_TEXTURES": "random\ntextures",
    "PROMPT_GEMINI": "diffusion\ntextures",
    "HUMAN_EVAL": "human\nbaseline",
}
HUMAN_EVAL_CSVS = {
    "Mugs": CSV_DIR / "l2_per_keypoint_human_eval_mugs.csv",
}


def load_flat_csv(path):
    """Return {model_label: [all_l2_errors]} from a CSV where columns are {MODEL}_{kp}."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        columns = {}
        for col in reader.fieldnames:
            m = re.search(r"_([a-z])", col)
            idx = m.start()
            columns[col] = (col[:idx], col[idx + 1 :])
        model_labels = list(dict.fromkeys(model for model, _ in columns.values()))
        errors = {label: [] for label in model_labels}
        for row in reader:
            for col, (model, _) in columns.items():
                if row[col]:
                    errors[model].append(float(row[col]))
    return errors


def load_csv(cat_name):
    """Return (model_labels, {model: [per_image_mean_l2]}, {(model, kp): [l2_errors]}).

    Each row in the CSV is one image. Per-image AKD is the mean across
    available keypoint columns for that model.
    """
    path = CSV_DIR / f"l2_per_keypoint_{cat_name.lower()}.csv"
    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        # Model names are ALL_CAPS, keypoint names are lowercase.
        # Split at the first "_" followed by a lowercase letter.
        columns = {}  # col_name -> (model, kp)
        for col in reader.fieldnames:
            m = re.search(r"_([a-z])", col)
            idx = m.start()
            columns[col] = (col[:idx], col[idx + 1 :])

        model_labels = list(dict.fromkeys(model for model, _ in columns.values()))
        kp_names = list(dict.fromkeys(kp for _, kp in columns.values()))
        model_cols = {label: [c for c, (m, _) in columns.items() if m == label] for label in model_labels}

        per_image = {label: [] for label in model_labels}
        per_kp = {(model, kp): [] for model in model_labels for kp in kp_names}
        for row in reader:
            for label, cols in model_cols.items():
                vals = [float(row[c]) for c in cols if row[c]]
                if vals:
                    per_image[label].append(np.mean(vals))
            for col, (model, kp) in columns.items():
                if row[col]:
                    per_kp[(model, kp)].append(float(row[col]))

    return model_labels, kp_names, per_image, per_kp


def draw_boxplot(ax_box, axes_box, box_data, all_box_labels, palette, cat_name):
    """Render the horizontal boxplot with fliers and mean markers."""
    display_labels = [LABEL_MAP.get(lbl, lbl) for lbl in all_box_labels]
    n_boxes = len(box_data)
    max_boxes = max(len(CATEGORIES), 4)
    positions = list(range(1, n_boxes + 1))
    all_display_labels = display_labels + [""] * (max_boxes - n_boxes)
    xlim_max = XLIMS[cat_name]

    ax_box.boxplot(
        box_data,
        positions=positions,
        patch_artist=True,
        boxprops=dict(alpha=0.7),
        medianprops=dict(color="black", linewidth=2),
        whis=1.5,
        showfliers=False,
        vert=False,
    )
    ax_box.set_yticks(range(1, max_boxes + 1))
    if ax_box == axes_box[0]:
        ax_box.set_yticklabels(all_display_labels)
    else:
        ax_box.set_yticklabels([""] * max_boxes)
    ax_box.set_xlim(left=-1, right=xlim_max)
    ax_box.set_ylim(0.5, max_boxes + 0.5)

    for j, (errors, color) in enumerate(zip(box_data, palette), start=1):
        q1, q3 = np.percentile(errors, 25), np.percentile(errors, 75)
        lo = q1 - 1.5 * (q3 - q1)
        hi = q3 + 1.5 * (q3 - q1)
        fliers = [e for e in errors if (e < lo or e > hi) and 0 <= e <= xlim_max]
        if fliers:
            ax_box.scatter(fliers, [j] * len(fliers), marker=".", s=4, alpha=0.4, color=color, zorder=2)
    for j, errors in enumerate(box_data, start=1):
        mean, median = np.mean(errors), np.median(errors)
        ax_box.scatter(mean, j, marker="D", color="white", edgecolors="black", zorder=3, s=30)
        ax_box.text(median, j + 0.3, f"{median:.1f}", ha="center", va="bottom", color=palette[j - 1])
    for patch, color in zip(ax_box.patches, palette):
        patch.set_facecolor(color)

    ax_box.set_title(cat_name)
    ax_box.set_xlabel("L2 (pixels)")
    ax_box.tick_params(axis="y", length=0)
    sns.despine(ax=ax_box, left=True)


def main():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.family": "serif",
        }
    )
    palette = sns.color_palette("tab10")
    sns.set_style("ticks")
    out_dir = CSV_DIR

    fig_hist, axes_hist = plt.subplots(1, 3, figsize=(12, 4))
    fig_cdf, axes_cdf = plt.subplots(1, 3, figsize=(12, 4))
    fig_box, axes_box = plt.subplots(1, 3, figsize=(15 / 2.54, 4.0))

    for ax_hist, ax_cdf, ax_box, cat_name in zip(axes_hist, axes_cdf, axes_box, CATEGORIES):
        model_labels, kp_names, per_image, per_kp = load_csv(cat_name)

        # optionally append human eval data
        extra_labels, extra_box_data = [], []
        if cat_name in HUMAN_EVAL_CSVS:
            he = load_flat_csv(HUMAN_EVAL_CSVS[cat_name])
            for lbl, errs in he.items():
                extra_labels.append(lbl)
                extra_box_data.append(errs)

        box_data = []
        for i, label in enumerate(model_labels):
            errors = [e for kp in kp_names for e in per_kp[(label, kp)]]
            plot_label = f"{label} (μ={np.mean(errors):.1f}, med={np.median(errors):.1f})"

            sns.histplot(
                errors,
                ax=ax_hist,
                label=plot_label,
                color=palette[i],
                kde=True,
                stat="density",
                alpha=0.4,
                binwidth=5,
            )

            sorted_errors = np.sort(errors)
            cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            ax_cdf.plot(sorted_errors, cdf, label=plot_label, color=palette[i])

            box_data.append(errors)

        all_box_labels = model_labels + extra_labels
        box_data = box_data + extra_box_data
        draw_boxplot(ax_box, axes_box, box_data, all_box_labels, palette, cat_name)

        for ax in (ax_hist, ax_cdf):
            ax.set_title(cat_name)
            ax.set_xlabel("AKD (pixels)")
            ax.legend(fontsize=8)
            ax.set_xlim(left=0)

        ax_hist.set_ylabel("Density")
        ax_cdf.set_ylabel("CDF")
        ax_cdf.axvline(x=32, color="gray", linestyle="--", linewidth=0.8)

    fig_hist.suptitle("Per-image AKD distributions")
    fig_hist.tight_layout()
    fig_hist.savefig(out_dir / "l2_distributions.png", dpi=150, bbox_inches="tight")
    print("Saved l2_distributions.png")

    fig_cdf.suptitle("Per-image AKD cumulative distribution")
    fig_cdf.tight_layout()
    fig_cdf.savefig(out_dir / "l2_cdf.png", dpi=150, bbox_inches="tight")
    print("Saved l2_cdf.png")

    fig_box.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    fig_box.savefig(out_dir / "l2_boxplot.png", bbox_inches="tight")
    print("Saved l2_boxplot.svg")


if __name__ == "__main__":
    main()
