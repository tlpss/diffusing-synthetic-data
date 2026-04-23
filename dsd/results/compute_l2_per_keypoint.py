"""
Compute per-keypoint L2 errors for mugs, shoes, tshirts and write CSVs.
Compares: real baseline, random textures, best diffusion.

Usage:
    conda run -n dsd python dsd/results/compute_l2_per_keypoint.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "keypoint-detection"))

import numpy as np
import torch
from experiments.keypoint_wandb_runs import (
    PROMPT_GEMINI_MUGS_RUN,
    PROMPT_GEMINI_SHOES_RUN,
    PROMPT_GEMINI_TSHIRTS_RUN,
    RANDOM_TEXTURES_MUGS_RUN,
    RANDOM_TEXTURES_SHOES_RUN,
    RANDOM_TEXTURES_TSHIRTS_RUN,
    REAL_MUGS_RUN,
    REAL_SHOES_RUN,
    REAL_TSHIRTS_RUN,
    get_model_checkpoint_artifact_from_run,
)
from experiments.paths import REAL_MUGS_TEST_DATASET, REAL_SHOES_TEST_DATASET, REAL_TSHIRTS_TEST_DATASET
from keypoint_detection.data.coco_dataset import COCOKeypointsDataset
from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap_batch_maxpool
from keypoint_detection.utils.load_checkpoints import get_model_from_wandb_checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

from dsd.generate_coco_datasets_from_diffusion_renders import mug_category, shoe_category, tshirt_category

CATEGORIES = {
    "Mugs": {
        "runs": {
            "REAL": REAL_MUGS_RUN,
            "RANDOM_TEXTURES": RANDOM_TEXTURES_MUGS_RUN,
            "PROMPT_GEMINI": PROMPT_GEMINI_MUGS_RUN,
        },
        "dataset": REAL_MUGS_TEST_DATASET,
        "channel_config": [[c] for c in mug_category.keypoints],
    },
    "Shoes": {
        "runs": {
            "REAL": REAL_SHOES_RUN,
            "RANDOM_TEXTURES": RANDOM_TEXTURES_SHOES_RUN,
            "PROMPT_GEMINI": PROMPT_GEMINI_SHOES_RUN,
        },
        "dataset": REAL_SHOES_TEST_DATASET,
        "channel_config": [[c] for c in shoe_category.keypoints],
    },
    "Tshirts": {
        "runs": {
            "REAL": REAL_TSHIRTS_RUN,
            "RANDOM_TEXTURES": RANDOM_TEXTURES_TSHIRTS_RUN,
            "PROMPT_GEMINI": PROMPT_GEMINI_TSHIRTS_RUN,
        },
        "dataset": REAL_TSHIRTS_TEST_DATASET,
        "channel_config": [[c] for c in tshirt_category.keypoints],
    },
}


def get_per_keypoint_errors(model, dataset_json_path, channel_config):
    """Returns {keypoint_name: [l2_errors]} across all images."""
    dataset = COCOKeypointsDataset(
        dataset_json_path,
        keypoint_channel_configuration=channel_config,
        detect_only_visible_keypoints=False,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    model.eval()
    model.cuda()

    # channel_config is [[kp_name], [kp_name], ...]
    kp_names = [ch[0] for ch in channel_config]
    per_kp_errors = {name: [] for name in kp_names}

    with torch.no_grad():
        for image, keypoints in tqdm(dataloader, leave=False):
            image = image.cuda()
            heatmaps = model(image)
            predicted_keypoints = get_keypoints_from_heatmap_batch_maxpool(heatmaps, max_keypoints=1)[0]

            for i, kp_name in enumerate(kp_names):
                if not predicted_keypoints[i] or len(keypoints[i]) == 0:
                    continue
                pred = torch.tensor(predicted_keypoints[i][0], dtype=torch.float32)
                gt = torch.tensor(keypoints[i][0], dtype=torch.float32)
                per_kp_errors[kp_name].append(torch.norm(pred - gt).item())

    return per_kp_errors


def main():
    import csv

    # all_errors[cat_name][label] = {kp_name: [l2_errors]}
    all_errors = {}

    for cat_name, cfg in CATEGORIES.items():
        print(f"\n=== {cat_name} ===")
        all_errors[cat_name] = {}
        for label, run_id in cfg["runs"].items():
            artifact = get_model_checkpoint_artifact_from_run(run_id)
            model = get_model_from_wandb_checkpoint(artifact).cuda()
            kp_errors = get_per_keypoint_errors(model, cfg["dataset"], cfg["channel_config"])
            all_errors[cat_name][label] = kp_errors
            for kp, errs in kp_errors.items():
                print(f"  {label:25s}  {kp}: mean={np.mean(errs):.1f}  median={np.median(errs):.1f}")
            del model
            torch.cuda.empty_cache()

    out_dir = Path(__file__).parent
    out_dir.mkdir(exist_ok=True)

    # --- CSV: one file per category, rows=keypoints, columns=model mean/median ---
    for cat_name, run_errors in all_errors.items():
        model_labels = list(run_errors.keys())
        kp_names = list(next(iter(run_errors.values())).keys())

        csv_path = out_dir / f"l2_per_keypoint_{cat_name.lower()}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            # columns: one per (model, keypoint), rows = per-image observations
            columns = [(label, kp) for label in model_labels for kp in kp_names]
            writer.writerow([f"{label}_{kp}" for label, kp in columns])
            n_rows = max(len(run_errors[label][kp]) for label, kp in columns)
            for i in range(n_rows):
                row = []
                for label, kp in columns:
                    errors = run_errors[label][kp]
                    row.append(f"{errors[i]:.2f}" if i < len(errors) else "")
                writer.writerow(row)
        print(f"Saved {csv_path.name}")


if __name__ == "__main__":
    main()
