"""
Compute PCK (Percentage of Correct Keypoints) from per-keypoint L2 CSVs,
using per-image bounding box sizes from the GT COCO annotations as normaliser.

PCK@k = fraction of keypoints where L2_error < k * bbox_diagonal
  - Mugs/Shoes: bbox diagonal = sqrt(w^2 + h^2) from the SAM-masked annotation
    'bbox' field (lab-mugs_sam_masked.json / annotations_sam_masked.json)
  - Tshirts: bbox diagonal from the 'bbox' field in the standard annotations

CSV rows are in annotation order (shuffle=False in the original dataloader),
so row i maps to the i-th COCO annotation.

Reads:  l2_per_keypoint_{mugs,shoes,tshirts}.csv
Writes: pck_summary.csv

Usage:
    conda run -n dsd python dsd/results/compute_pck_from_csv.py
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

CSV_DIR = Path(__file__).parent
REPO_ROOT = CSV_DIR.parents[1]

# COCO annotation paths (one annotation per image, in image order).
# SAM-masked variants are used for Mugs and Shoes as they include 'bbox' fields
# derived from segmentation masks, giving a better bbox normaliser than keypoint extents.
ANNOTATION_PATHS = {
    "Mugs": REPO_ROOT / "data/real/mugs/lab-mugs_resized_512x512/lab-mugs_sam_masked.json",
    "Shoes": REPO_ROOT / "data/real/shoes/dsd-shoes-real_resized_512x512/annotations_sam_masked.json",
    "Tshirts": REPO_ROOT / "data/real/tshirts/tshirts-test-kpcentercropped/annotations.json",
}

PCK_THRESHOLDS = [0.05, 0.1, 0.15, 0.2]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_bbox_diagonals(annotation_path: Path) -> list[float]:
    """
    Return a list of per-image bbox diagonals in annotation order.

    If the annotation has a 'bbox' field (COCO [x, y, w, h]) use it;
    otherwise fall back to the extent of the visible GT keypoints.
    """
    with open(annotation_path) as f:
        data = json.load(f)

    # sort by image_id to match COCOKeypointsDataset iteration order
    annotations = sorted(data["annotations"], key=lambda a: a["image_id"])

    diagonals = []
    for ann in annotations:
        if "bbox" in ann:
            _, _, w, h = ann["bbox"]
            diagonals.append((w**2 + h**2) ** 0.5)
        else:
            kps = ann["keypoints"]
            xs = kps[0::3]
            ys = kps[1::3]
            vs = kps[2::3]
            vis_x = [x for x, v in zip(xs, vs) if v > 0]
            vis_y = [y for y, v in zip(ys, vs) if v > 0]
            if len(vis_x) >= 2:
                diagonals.append(max(max(vis_x) - min(vis_x), max(vis_y) - min(vis_y)))
            else:
                diagonals.append(float("nan"))

    return diagonals


def load_csv(cat_name: str):
    """
    Return (model_labels, kp_names, per_kp) where
    per_kp[(model, kp)] = list of (l2_error, row_index) keeping row order.
    """
    path = CSV_DIR / f"l2_per_keypoint_{cat_name.lower()}.csv"
    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        columns = {}
        for col in reader.fieldnames:
            m = re.search(r"_([a-z])", col)
            idx = m.start()
            columns[col] = (col[:idx], col[idx + 1 :])

        model_labels = list(dict.fromkeys(model for model, _ in columns.values()))
        kp_names = list(dict.fromkeys(kp for _, kp in columns.values()))

        # per_kp[(model, kp)] = list of (value, row_idx)
        per_kp = {(model, kp): [] for model in model_labels for kp in kp_names}
        for row_idx, row in enumerate(reader):
            for col, (model, kp) in columns.items():
                if row[col]:
                    per_kp[(model, kp)].append((float(row[col]), row_idx))

    return model_labels, kp_names, per_kp


def pck_bbox(observations: list[tuple[float, int]], bbox_diagonals: list[float], k: float) -> float:
    """
    PCK@k using per-image bbox diagonal as normaliser.
    observations: list of (l2_error, row_idx)
    """
    if not observations:
        return float("nan")
    correct = sum(
        1 for l2, row_idx in observations if not np.isnan(bbox_diagonals[row_idx]) and l2 < k * bbox_diagonals[row_idx]
    )
    total = sum(1 for _, row_idx in observations if not np.isnan(bbox_diagonals[row_idx]))
    return correct / total if total > 0 else float("nan")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    summary_rows = []

    for cat_name in ["Mugs", "Shoes", "Tshirts"]:
        bbox_diagonals = load_bbox_diagonals(ANNOTATION_PATHS[cat_name])
        model_labels, kp_names, per_kp = load_csv(cat_name)

        print(f"\n=== {cat_name} ===")
        print(
            f"  bbox diagonal: mean={np.nanmean(bbox_diagonals):.1f}  "
            f"median={np.nanmedian(bbox_diagonals):.1f}  "
            f"(pixel thresholds at k=0.1: {0.1 * np.nanmedian(bbox_diagonals):.1f} px)"
        )

        # --- per-keypoint breakdown ---
        header = ["keypoint", "model"] + [f"PCK@{k:.2f}" for k in PCK_THRESHOLDS]
        print("  " + "  ".join(header))

        for kp in kp_names:
            for label in model_labels:
                obs = per_kp[(label, kp)]
                vals = [pck_bbox(obs, bbox_diagonals, k) for k in PCK_THRESHOLDS]
                print(f"  {kp:30s}  {label:25s}  " + "  ".join(f"{v:.3f}" for v in vals))

        # --- per-model overall (all keypoints pooled) ---
        print(f"\n  --- {cat_name} overall ---")
        for label in model_labels:
            all_obs = [item for kp in kp_names for item in per_kp[(label, kp)]]
            vals = [pck_bbox(all_obs, bbox_diagonals, k) for k in PCK_THRESHOLDS]
            print(f"  {label:25s}  " + "  ".join(f"PCK@{k:.2f}={v:.3f}" for k, v in zip(PCK_THRESHOLDS, vals)))
            summary_rows.append([cat_name, label] + vals)

    # save summary CSV
    summary_path = CSV_DIR / "pck_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "model"] + [f"PCK@{k:.2f}" for k in PCK_THRESHOLDS])
        writer.writerows(summary_rows)
    print(f"\nSaved {summary_path.name}")


if __name__ == "__main__":
    main()
