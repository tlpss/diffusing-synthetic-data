"""
proxies for measuring the semantic similarities between the original image (for which we have labels) and the diffusion rendered image (for which we do not have labels).
"""
import pathlib
from urllib.request import urlretrieve

import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

from dsd import MODEL_CACHE

VIT_H_CHECKPOINT_PATH = "vit_h_4b8939.pth"
VIT_H_REMOTE_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

VIT_B_CHECKPOINT_PATH = "vit_b_01ec64.pth"
VIT_B_REMOTE_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

CHECKPOINT_CACHE = MODEL_CACHE / "sam"
CHECKPOINT_CACHE.mkdir(parents=True, exist_ok=True)


def load_sam_model(model_type: str = "vit_b"):
    if model_type == "vit_h":
        checkpoint_path = CHECKPOINT_CACHE / VIT_H_CHECKPOINT_PATH
        if not checkpoint_path.exists():
            urlretrieve(VIT_B_REMOTE_CHECKPOINT_URL, checkpoint_path)
    elif model_type == "vit_b":
        checkpoint_path = CHECKPOINT_CACHE / VIT_B_CHECKPOINT_PATH
        if not checkpoint_path.exists():
            urlretrieve(VIT_B_REMOTE_CHECKPOINT_URL, checkpoint_path)
    else:
        raise ValueError(f"model type {model_type} not supported")

    sam_model = sam_model_registry[model_type](checkpoint_path).cuda()
    sam_predictor = SamPredictor(sam_model)
    return sam_predictor


def calculate_mask_IoU(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    calculate the IoU between two masks
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


@torch.no_grad()
def calculate_similarities_for_render_dir(render_dir: pathlib.Path, sam_predictor: SamPredictor):
    original_mask = render_dir / "original" / "segmentation.png"
    # get bounding box for the binary mask (0 is background, 1 is foreground)
    original_mask = np.array(Image.open(original_mask))
    original_mask = original_mask > 0.5
    # find min max x and y
    foreground_indices = np.where(original_mask == True)  # noqa
    min_x = foreground_indices[1].min()
    max_x = foreground_indices[1].max()
    min_y = foreground_indices[0].min()
    max_y = foreground_indices[0].max()

    # add a bit of margin to the bounding box
    # this results in better SAM masks!
    margin = 10
    min_x = max(0, min_x - margin)
    min_y = max(0, min_y - margin)
    max_x = min(original_mask.shape[1], max_x + margin)
    max_y = min(original_mask.shape[0], max_y + margin)

    bbox = (min_x, min_y, max_x, max_y)

    iou_scores_dict = {}
    # get subdirs
    subdirs = [d for d in render_dir.iterdir() if d.is_dir()]
    # subdirs = [d for d in subdirs if d.name != "original"]
    for i, subdir in enumerate(subdirs):
        images = list(subdir.glob("*.png"))
        if subdir.name == "original":
            images = [d for d in images if "rgb" in d.name]

        for image_path in images:
            image = np.array(Image.open(image_path))
            # check if image is all black -> NSFW filter -> skip
            if np.max(image) < 1e-4:
                print(f"skipping {image_path} because it is all black")
                continue
            sam_predictor.set_image(image)
            masks, _, _ = sam_predictor.predict(box=np.array(bbox), multimask_output=True)
            mask = masks[0]

            # # save mask
            # print(mask)
            # mask = np.concatenate([mask[...,None], mask[...,None], mask[...,None]], axis=-1)
            # mask = mask * 255.0
            # mask = mask.astype(np.uint8)
            # mask = Image.fromarray(mask)

            # mask.save("mask.png")
            # # save image
            # image = Image.fromarray(image)
            # # add bbox on image
            # from PIL import ImageDraw
            # draw = ImageDraw.Draw(image)
            # draw.rectangle((min_x,min_y,max_x,max_y), outline='red')

            # # blend image with mask
            # image = Image.blend(image, mask, alpha=0.5)

            # image.save("image.png")

            # calculate IoU
            mask_iou = calculate_mask_IoU(original_mask, mask)
            # print(f"iou score for {image_path} is {mask_iou}")
            iou_scores_dict[str(image_path)] = mask_iou
    return iou_scores_dict


def calculate_similarities_for_renders(render_path: pathlib.Path, sam_predictor: SamPredictor):
    render_paths = list(render_path.glob("**/rgb.png"))
    render_dirs = [p.parents[1] for p in render_paths]

    similarity_dict = {}
    for render_dir in tqdm(render_dirs):
        render_dir_simialrity_dict = calculate_similarities_for_render_dir(render_dir, sam_predictor)
        similarity_dict.update(render_dir_simialrity_dict)
    return similarity_dict


if __name__ == "__main__":
    from dsd import DATA_DIR

    sam_predictor = load_sam_model()

    render_dataset = DATA_DIR / "diffusion_renders" / "mugs" / "run_3"
    similarity_dict = calculate_similarities_for_renders(render_dataset, sam_predictor)

    # set the dict keys to relative paths
    similarity_dict = {str(pathlib.Path(k).relative_to(render_dataset)): v for k, v in similarity_dict.items()}

    # dict to csv
    import pandas as pd

    df = pd.DataFrame.from_dict(similarity_dict, orient="index", columns=["sam_IoU"])
    # convert path to proper column and add new index
    df = df.reset_index()
    df = df.rename(columns={"index": "relative_path"})
    df.to_csv(str(render_dataset / "sam_IoUs.csv"))
