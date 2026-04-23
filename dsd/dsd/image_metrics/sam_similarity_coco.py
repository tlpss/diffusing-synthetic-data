"""
Compute SAM IoU similarity using a COCO-formatted dataset.

For each image, the COCO segmentation mask is used as ground truth and as a
bounding-box prompt for SAM.  The IoU between the SAM-predicted mask and the
ground-truth mask is recorded.

Results are written to sam_IoUs.csv next to the annotations.json.
"""
import pathlib

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

from dsd.image_metrics.sam_similarity_diffusion_rendered import calculate_mask_IoU, load_sam_model


def coco_rle_to_mask(segmentation: dict) -> np.ndarray:
    return mask_utils.decode(segmentation).astype(bool)


def coco_bbox_to_xyxy(bbox):
    """Convert COCO [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


@torch.no_grad()
def compute_sam_ious_for_coco(annotations_path: pathlib.Path, model_type: str = "vit_b"):
    import json

    annotations_path = pathlib.Path(annotations_path)
    dataset_dir = annotations_path.parent

    data = json.loads(annotations_path.read_text())
    id_to_image = {img["id"]: img for img in data["images"]}

    sam_predictor = load_sam_model(model_type)

    iou_scores = {}
    for ann in tqdm(data["annotations"]):
        image_info = id_to_image[ann["image_id"]]
        image_path = dataset_dir / image_info["file_name"]

        if not image_path.exists():
            print(f"skipping missing image: {image_path}")
            continue

        image = np.array(Image.open(image_path).convert("RGB"))
        if np.max(image) < 1e-4:
            print(f"skipping black image: {image_path}")
            continue

        gt_mask = coco_rle_to_mask(ann["segmentation"])
        bbox = np.array(coco_bbox_to_xyxy(ann["bbox"]))

        sam_predictor.set_image(image)
        masks, _, _ = sam_predictor.predict(box=bbox, multimask_output=True)
        sam_mask = masks[0]

        iou = calculate_mask_IoU(gt_mask, sam_mask)
        iou_scores[image_info["file_name"]] = iou

    return iou_scores


if __name__ == "__main__":
    pass

    import click
    import pandas as pd

    @click.command()
    @click.option("--annotations", type=pathlib.Path, required=True, help="Path to COCO annotations.json")
    @click.option("--model_type", default="vit_b", show_default=True)
    def main(annotations: pathlib.Path, model_type: str):
        iou_scores = compute_sam_ious_for_coco(annotations, model_type)

        df = pd.DataFrame.from_dict(iou_scores, orient="index", columns=["sam_IoU"])
        df = df.reset_index().rename(columns={"index": "file_name"})

        out_path = annotations.parent / "sam_IoUs.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} scores to {out_path}")

    main()
