"""
Compute per-keypoint L2 errors for the human eval on mugs and write a CSV.

The output format matches l2_per_keypoint_mugs.csv so the human-eval column
can be compared side-by-side with the model columns.

Usage:
    conda run -n dsd python dsd/results/compute_l2_per_keypoint_human_eval.py
"""

import csv
import json
import math
from pathlib import Path

HUMAN_EVAL_JSON = Path(__file__).parents[2] / "data/real/mugs-lab-human-eval/annotations.json"
GT_JSON = Path(__file__).parents[2] / "data/real/mugs/lab-mugs_resized_512x512/lab-mugs.json"

KEYPOINT_NAMES = ["bottom", "handle", "top"]


def parse_keypoints(kp_flat):
    """Return list of (x, y) or None per keypoint (visibility 0 → None)."""
    result = []
    for i in range(0, len(kp_flat), 3):
        x, y, v = kp_flat[i], kp_flat[i + 1], kp_flat[i + 2]
        result.append((x, y) if v > 0 else None)
    return result


def main():
    he = json.load(open(HUMAN_EVAL_JSON))
    gt = json.load(open(GT_JSON))

    he_img_id_to_file = {img["id"]: img["file_name"].split("/")[-1] for img in he["images"]}
    gt_file_to_img_id = {img["file_name"].split("/")[-1]: img["id"] for img in gt["images"]}

    he_imgid_to_ann = {ann["image_id"]: ann for ann in he["annotations"]}
    gt_imgid_to_ann = {ann["image_id"]: ann for ann in gt["annotations"]}

    # per_kp_errors[kp_name] = [l2_error, ...] one entry per matched image
    per_kp_errors = {name: [] for name in KEYPOINT_NAMES}

    matched = 0
    for he_img_id, filename in he_img_id_to_file.items():
        gt_img_id = gt_file_to_img_id.get(filename)
        if gt_img_id is None:
            continue
        he_ann = he_imgid_to_ann.get(he_img_id)
        gt_ann = gt_imgid_to_ann.get(gt_img_id)
        if he_ann is None or gt_ann is None:
            continue

        pred_kps = parse_keypoints(he_ann["keypoints"])
        gt_kps = parse_keypoints(gt_ann["keypoints"])

        for ch, name in enumerate(KEYPOINT_NAMES):
            pred_kp = pred_kps[ch] if ch < len(pred_kps) else None
            gt_kp = gt_kps[ch] if ch < len(gt_kps) else None
            if pred_kp is None or gt_kp is None:
                continue
            dist = math.sqrt((pred_kp[0] - gt_kp[0]) ** 2 + (pred_kp[1] - gt_kp[1]) ** 2)
            per_kp_errors[name].append(dist)

        matched += 1

    print(f"Matched images: {matched}")
    for name in KEYPOINT_NAMES:
        errs = per_kp_errors[name]
        if errs:
            import statistics

            print(f"  {name}: mean={sum(errs)/len(errs):.2f}  median={statistics.median(errs):.2f}  n={len(errs)}")

    out_path = Path(__file__).parent / "l2_per_keypoint_human_eval_mugs.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"HUMAN_EVAL_{name}" for name in KEYPOINT_NAMES])
        n_rows = max(len(per_kp_errors[name]) for name in KEYPOINT_NAMES)
        for i in range(n_rows):
            row = []
            for name in KEYPOINT_NAMES:
                errs = per_kp_errors[name]
                row.append(f"{errs[i]:.2f}" if i < len(errs) else "")
            writer.writerow(row)

    print(f"Saved {out_path.name}")


if __name__ == "__main__":
    main()
