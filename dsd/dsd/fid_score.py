from torchmetrics.image import FrechetInceptionDistance

"""
measure similarity between the diffused image and the target caption using CLIP Score
https://arxiv.org/pdf/2104.08718.pdf

"""
import pathlib
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def get_images_per_renderer(render_path: pathlib.Path):
    # get all files in the folder
    renderer_to_images_dict = defaultdict(list)
    image_paths = list(render_path.glob("**/*.png"))
    # remove all images with "rgb.png", "depth_image.png" ,"segmentation.png"
    image_paths = [x for x in image_paths if not str(x.parents[0]).endswith("original") and x.suffix == ".png"]
    for path in image_paths:
        renderer = path.parent.stem
        renderer_to_images_dict[str(renderer)].append(str(path))
    return renderer_to_images_dict


def get_tensor_from_image_path(image_path):
    image = np.array(Image.open(image_path))
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    image = image / 255.0
    image = image.unsqueeze(0)
    return image


class FIDScoreCalculator:
    """FID score calculation for a dsd dataset with respect to a real dataset

    uses torchmetrics implemnetation
    """

    def __init__(self):
        self._metric = FrechetInceptionDistance(normalize=True).to("cuda")

    @torch.no_grad()
    def calculate_scores(self, render_path: pathlib.Path, real_path: pathlib.Path):
        renderer_to_images_dict = get_images_per_renderer(render_path)

        real_images = list(real_path.glob("**/*.jpg"))

        fid_score_dict = {}
        for renderer, images in tqdm(renderer_to_images_dict.items()):
            self._metric.reset()
            for img in real_images:
                img = get_tensor_from_image_path(img).to("cuda")
                self._metric.update(img, real=True)
            for img in images:
                img = get_tensor_from_image_path(img).to("cuda")
                self._metric.update(img, real=False)

            fid_score = self._metric.compute()
            fid_score_dict[renderer] = fid_score.item()

        return fid_score_dict


if __name__ == "__main__":
    from dsd import DATA_DIR

    render_dataset = DATA_DIR / "diffusion_renders" / "mugs" / "run_3"
    real_path = DATA_DIR / "real" / "mugs" / "lab-mugs"

    fid_scorer = FIDScoreCalculator()
    similarity_dict = fid_scorer.calculate_scores(render_dataset, real_path)
    print(similarity_dict)

    # dict to csv
    import pandas as pd

    df = pd.DataFrame.from_dict(similarity_dict, orient="index", columns=["fid-score"])
    # convert path to proper column and add new index
    df = df.reset_index()
    df = df.rename(columns={"index": "renderer"})
    df.to_csv(str(render_dataset / "fid_scores.csv"))
