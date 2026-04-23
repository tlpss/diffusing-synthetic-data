"""
Compute mAP and AKD for the human-eval keypoint annotations
compared to the ground-truth lab-mugs_resized_512x512 test set.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "keypoint-detection"))

from keypoint_detection.models.metrics import DetectedKeypoint, Keypoint, KeypointAPMetrics

HUMAN_EVAL_JSON = Path(__file__).parents[2] / "data/real/mugs-lab-human-eval/annotations.json"
GT_JSON = Path(__file__).parents[2] / "data/real/mugs/lab-mugs_resized_512x512/lab-mugs.json"

KEYPOINT_NAMES = ["bottom", "handle", "top"]
THRESHOLD_DISTANCES = [8, 16, 32]


def parse_keypoints(kp_flat):
    """Return list of (x, y) or None per keypoint (visibility 0 → None)."""
    result = []
    for i in range(0, len(kp_flat), 3):
        x, y, v = kp_flat[i], kp_flat[i + 1], kp_flat[i + 2]
        result.append((x, y) if v > 0 else None)
    return result


def compute_keypoint_metrics(predictions, ground_truths, keypoint_names, threshold_distances):
    """
    Compute AKD, median KD, and mAP from matched prediction/GT keypoint lists.

    predictions, ground_truths: list of per-image lists of (x, y) or None per keypoint.

    Returns mae_dict, median_dict, meanAP — keyed by keypoint name, matching the
    format produced by generate_run_metrics.generate_AKD_dict / generate_meanAP_dict.
    """
    ap_metrics = [KeypointAPMetrics(threshold_distances) for _ in keypoint_names]
    errors = [[] for _ in keypoint_names]

    for pred_kps, gt_kps in zip(predictions, ground_truths):
        for ch, (pred_kp, gt_kp) in enumerate(zip(pred_kps, gt_kps)):
            if pred_kp is None or gt_kp is None:
                continue
            dist = math.sqrt((pred_kp[0] - gt_kp[0]) ** 2 + (pred_kp[1] - gt_kp[1]) ** 2)
            errors[ch].append(dist)
            ap_metrics[ch].update(
                [DetectedKeypoint(u=pred_kp[0], v=pred_kp[1], probability=1.0)],
                [Keypoint(u=gt_kp[0], v=gt_kp[1])],
            )

    mae_dict = {name: sum(errors[ch]) / len(errors[ch]) for ch, name in enumerate(keypoint_names) if errors[ch]}
    median_dict = {name: np.median(errors[ch]).item() for ch, name in enumerate(keypoint_names) if errors[ch]}
    meanAP_dict = {
        name: sum(ap_metrics[ch].compute().values()) / len(threshold_distances)
        for ch, name in enumerate(keypoint_names)
    }

    return mae_dict, median_dict, meanAP_dict


def main():
    he = json.load(open(HUMAN_EVAL_JSON))
    gt = json.load(open(GT_JSON))

    # index by filename stem
    he_img_id_to_file = {img["id"]: img["file_name"].split("/")[-1] for img in he["images"]}
    gt_file_to_img_id = {img["file_name"].split("/")[-1]: img["id"] for img in gt["images"]}

    he_imgid_to_ann = {ann["image_id"]: ann for ann in he["annotations"]}
    gt_imgid_to_ann = {ann["image_id"]: ann for ann in gt["annotations"]}

    predictions, ground_truths = [], []
    matched = 0
    for he_img_id, filename in he_img_id_to_file.items():
        gt_img_id = gt_file_to_img_id.get(filename)
        if gt_img_id is None:
            continue
        he_ann = he_imgid_to_ann.get(he_img_id)
        gt_ann = gt_imgid_to_ann.get(gt_img_id)
        if he_ann is None or gt_ann is None:
            continue
        matched += 1
        predictions.append(parse_keypoints(he_ann["keypoints"]))
        ground_truths.append(parse_keypoints(gt_ann["keypoints"]))

    print(f"Matched images: {matched}\n")

    mae_dict, median_dict, meanAP_dict = compute_keypoint_metrics(
        predictions, ground_truths, KEYPOINT_NAMES, THRESHOLD_DISTANCES
    )

    print("=== AKD (Average Keypoint Distance, pixels) ===")
    for name in KEYPOINT_NAMES:
        print(f"  {name}: {mae_dict[name]:.2f}")
    print(f"  overall: {sum(mae_dict.values()) / len(mae_dict):.2f}\n")

    print("=== Median Keypoint Distance (pixels) ===")
    for name in KEYPOINT_NAMES:
        print(f"  {name}: {median_dict[name]:.2f}")
    print(f"  overall: {sum(median_dict.values()) / len(median_dict):.2f}\n")

    print(f"=== mAP (distance thresholds: {THRESHOLD_DISTANCES} px) ===")
    for name in KEYPOINT_NAMES:
        print(f"  {name}: {meanAP_dict[name]:.3f}")
    print(f"  overall: {sum(meanAP_dict.values()) / len(meanAP_dict):.3f}")


if __name__ == "__main__":
    main()
