#!/usr/bin/env python3
"""
Creates a 2x3 grid showing predicted keypoints, ground truth keypoints,
and predicted segmentation masks for 2 random images per category
(mugs, shoes, tshirts).

  Rows    = 2 random images per category
  Columns = mugs | shoes | tshirts
  Each cell overlays:
    - Cyan contour:   predicted segmentation mask
    - Filled circles: predicted keypoints  (color per channel)
    - X crosses:      ground truth keypoints (same colors)
"""

import json
import random
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import torch
import wandb
from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap_batch_maxpool
from keypoint_detection.utils.load_checkpoints import get_model_from_wandb_checkpoint
from PIL import Image, ImageDraw
from ultralytics import YOLO

# ─── Config ───────────────────────────────────────────────────────────────────

SEED = 42
N_IMAGES = 2
OUTPUT_PATH = Path(__file__).parent / "detection_grid.png"

# ─── Copied verbatim from visualize_detections.ipynb ─────────────────────────

# https://sashamaps.net/docs/resources/20-colors/
DISTINCT_COLORS = [
    "#e6194B",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9A6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#ffffff",
    "#000000",
]


def get_model_artifact_name_from_run_id(run_id, artifact_type="model"):
    api = wandb.Api()
    run = api.run(run_id)
    logged_artifacts = run.logged_artifacts()
    model_artifact_names = []
    for artifact in logged_artifacts:
        if artifact.type == artifact_type:
            model_artifact_names.append(artifact.name)
            print(f"Found artifact: {artifact.name} (type: {artifact.type})")
    return model_artifact_names[0] if model_artifact_names else None


def get_segmentation_model_from_wandb_checkpoint(checkpoint_reference):
    if wandb.run is None:
        run = wandb.init(project="inference")
    else:
        run = wandb.run
    artifact = run.use_artifact(checkpoint_reference, type="model")
    artifact_dir = artifact.download()
    from pathlib import Path

    print("Files in artifact directory:")
    for file in Path(artifact_dir).iterdir():
        print(file)
    checkpoint_path = Path(artifact_dir) / "best.pt"
    return load_seg_model_from_checkpoint(checkpoint_path)


def load_seg_model_from_checkpoint(checkpoint_path):
    model = YOLO(checkpoint_path)
    return model


def get_seg_model_from_run(run_id):
    artifact = get_model_artifact_name_from_run_id(run_id)
    artifact_path = f"tlips/dsd-paper-yolo/{artifact}"
    model = get_segmentation_model_from_wandb_checkpoint(artifact_path)
    return model


def get_kp_model_from_run(run_id):
    artifact = get_model_artifact_name_from_run_id(run_id)
    artifact_path = f"tlips/dsd-paper/{artifact}"
    model = get_model_from_wandb_checkpoint(artifact_path)
    return model


def get_keypoints(image_path, model, device="cuda", abs_confidence_threshold=0.1):
    """Inference on a single image. Returns a list of [x, y] per channel."""
    model.eval().to(device)
    image = Image.open(image_path).convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        heatmaps = model(image).squeeze(0)
    predicted_keypoints = get_keypoints_from_heatmap_batch_maxpool(
        heatmaps.unsqueeze(0), abs_max_threshold=abs_confidence_threshold, max_keypoints=1
    )[0]
    formatted_keypoints = []
    for channel_keypoints in predicted_keypoints:
        if len(channel_keypoints) > 0:
            formatted_keypoints.append(channel_keypoints[0])
        else:
            formatted_keypoints.append(None)
    return formatted_keypoints


def get_segmentation_mask(image_path, model):
    """Returns the segmentation mask as a uint8 numpy array (H, W) at image resolution."""
    results = model.predict(source=image_path, save=False, verbose=False)
    if results[0].masks is None:
        raise ValueError("No segmentation mask found in the results.")
    segmask = results[0].masks.data.cpu().numpy()
    segmask = segmask[0] * 255
    # resize to original image size if the model processed at a different resolution
    orig_h, orig_w = results[0].orig_shape
    if segmask.shape != (orig_h, orig_w):
        import cv2

        segmask = cv2.resize(segmask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return segmask


def draw_seg_mask_contour(image, segmask, color=(255, 0, 0), thickness=1):
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    contours = skimage.measure.find_contours(segmask, 0.5)
    for contour in contours:
        contour = np.array(contour, dtype=np.int32)
        # flip x,y because skimage uses (row, col) while PIL uses (x, y)
        contour[:, [0, 1]] = contour[:, [1, 0]]
        draw.line(list(map(tuple, contour)), fill=color, width=thickness)
    return image


def draw_keypoints(image, keypoints, radius=5):
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    for i in range(len(keypoints)):
        if keypoints[i] is None:
            continue
        if len(keypoints[i]) != 2:
            print(f"Skipping keypoint {i} with invalid shape: {keypoints[i]}")
            continue
        x, y = keypoints[i]
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=DISTINCT_COLORS[i % len(DISTINCT_COLORS)],
        )
    return image


# ─── GT keypoints (not in notebook) ──────────────────────────────────────────


def draw_gt_keypoints(image, keypoints, radius=7):
    """Draw GT keypoints as X crosses with the same channel colors."""
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    for i, kp in enumerate(keypoints):
        if kp is None:
            continue
        x, y = float(kp[0]), float(kp[1])
        color = DISTINCT_COLORS[i % len(DISTINCT_COLORS)]
        s = radius
        draw.line([(x - s, y - s), (x + s, y + s)], fill=color, width=3)
        draw.line([(x - s, y + s), (x + s, y - s)], fill=color, width=3)
    return image


def parse_gt_keypoints(annotation):
    """Return list of (x, y) per keypoint; None when visibility == 0."""
    flat = annotation["keypoints"]
    n = len(flat) // 3
    return [(flat[3 * i], flat[3 * i + 1]) if flat[3 * i + 2] > 0 else None for i in range(n)]


# ─── Dataset configs ──────────────────────────────────────────────────────────

BASE = Path("/home/tlips/Documents/diffusing-synthetic-data/data/real")

CATEGORIES = {
    "mugs": {
        "data_dir": BASE / "mugs/lab-mugs_resized_512x512",
        "ann_file": "lab-mugs_sam_masked.json",
        "kp_run": "tlips/dsd-paper/runs/s5gwjor3",  # KP_PROMPT_GEMINI
        "seg_run": "tlips/dsd-paper-yolo/runs/nl4pqdu1",  # SEG_PROMPT_GEMINI
    },
    "shoes": {
        "data_dir": BASE / "shoes/dsd-shoes-real_resized_512x512",
        "ann_file": "annotations_sam_masked.json",
        "kp_run": "tlips/dsd-paper/runs/dl6ksv7b",  # KP_PROMPT_GEMINI
        "seg_run": "tlips/dsd-paper-yolo/runs/75dm6fa5",  # SEG_PROMPT_GEMINI
    },
    "tshirts": {
        "data_dir": BASE / "tshirts/tshirts-test-kpcentercropped",
        "ann_file": "annotations.json",
        "kp_run": "tlips/dsd-paper/runs/29zt96n2",  # KP_PROMPT_GEMINI
        "seg_run": "tlips/dsd-paper-yolo/runs/5v57ni2p",  # SEG_PROMPT_GEMINI
    },
}

CONTOUR_COLOR = (0, 255, 255)  # cyan
CONTOUR_THICKNESS = 5
KEYPOINT_RADIUS = 8

# ─── Data helpers ─────────────────────────────────────────────────────────────


def load_coco(data_dir, ann_file):
    with open(data_dir / ann_file) as f:
        return json.load(f)


def sample_annotated_images(coco, n, rng):
    """Return n (image_info, annotation) pairs sampled without replacement."""
    ann_by_image = {}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        if iid not in ann_by_image:
            ann_by_image[iid] = ann
    candidates = [(img, ann_by_image[img["id"]]) for img in coco["images"] if img["id"] in ann_by_image]
    return rng.sample(candidates, n)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    rng = random.Random(SEED)

    if wandb.run is None:
        wandb.init(project="visualization", reinit=True)

    cat_names = list(CATEGORIES.keys())
    fig, axes = plt.subplots(N_IMAGES, len(cat_names), figsize=(5 * len(cat_names), 5 * N_IMAGES))

    for col, cat_name in enumerate(cat_names):
        cfg = CATEGORIES[cat_name]
        print(f"\n=== {cat_name} ===")

        kp_model = get_kp_model_from_run(cfg["kp_run"])
        seg_model = get_seg_model_from_run(cfg["seg_run"])

        coco = load_coco(cfg["data_dir"], cfg["ann_file"])
        samples = sample_annotated_images(coco, N_IMAGES, rng)

        for row, (img_info, annotation) in enumerate(samples):
            img_path = cfg["data_dir"] / img_info["file_name"]
            print(f"  {img_path.name}")

            segmask = get_segmentation_mask(str(img_path), seg_model)
            pred_kps = get_keypoints(str(img_path), kp_model)
            gt_kps = parse_gt_keypoints(annotation)

            vis = Image.open(img_path).convert("RGB")
            vis = draw_seg_mask_contour(vis, segmask, color=CONTOUR_COLOR, thickness=CONTOUR_THICKNESS)
            vis = draw_gt_keypoints(vis, gt_kps, radius=KEYPOINT_RADIUS + 2)
            vis = draw_keypoints(vis, pred_kps, radius=KEYPOINT_RADIUS)

            ax = axes[row, col]
            ax.imshow(vis)
            ax.axis("off")
            if row == 0:
                ax.set_title(cat_name.capitalize(), fontsize=14, fontweight="bold", pad=6)

    legend_elements = [
        mpatches.Patch(facecolor="cyan", label="Predicted mask (contour)"),
        mpatches.Patch(facecolor="#aaa", label="Predicted KP (filled circle)"),
        mpatches.Patch(facecolor="#aaa", label="GT KP (cross ×)"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=3, fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, 0.0)
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
