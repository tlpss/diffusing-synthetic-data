"""
Compute CLIP score for a COCO-formatted diffusion dataset.

The text prompt is extracted from the image filename stem by splitting on '_'
and taking everything before the final index suffix (e.g. 'prompt text_0.png'
→ 'prompt text').

Results are written to clip_scores.csv next to the annotations.json.
"""
import pathlib

import click
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchmetrics.multimodal import CLIPScore
from tqdm import tqdm


def extract_caption(file_name: str) -> str:
    stem = pathlib.Path(file_name).stem
    # filename format: '<prompt text>_<index>'
    return "_".join(stem.split("_")[:-1])


@torch.no_grad()
def compute_clip_scores_for_coco(annotations_path: pathlib.Path, clip_model: str = "openai/clip-vit-base-patch16"):
    import json

    annotations_path = pathlib.Path(annotations_path)
    dataset_dir = annotations_path.parent

    data = json.loads(annotations_path.read_text())
    metric = CLIPScore(clip_model).to("cuda")

    scores = {}
    for img_info in tqdm(data["images"]):
        image_path = dataset_dir / img_info["file_name"]
        if not image_path.exists():
            print(f"skipping missing: {image_path}")
            continue

        image = np.array(Image.open(image_path).convert("RGB"))
        if np.max(image) < 1e-4:
            print(f"skipping black image: {image_path}")
            continue

        caption = extract_caption(img_info["file_name"])
        if not caption:
            print(f"skipping: no caption extractable from {img_info['file_name']}")
            continue

        img_tensor = torch.tensor(image).permute(2, 0, 1).to("cuda")
        score = metric([img_tensor], [caption])
        scores[img_info["file_name"]] = score.item()

    return scores


@click.command()
@click.option("--annotations", type=pathlib.Path, required=True, help="Path to COCO annotations.json")
@click.option("--clip_model", default="openai/clip-vit-base-patch16", show_default=True)
def main(annotations: pathlib.Path, clip_model: str):
    scores = compute_clip_scores_for_coco(annotations, clip_model)

    df = pd.DataFrame.from_dict(scores, orient="index", columns=["clip_score"])
    df = df.reset_index().rename(columns={"index": "file_name"})

    out_path = annotations.parent / "clip_scores.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} scores to {out_path}")


if __name__ == "__main__":
    main()
