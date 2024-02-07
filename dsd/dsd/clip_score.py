"""
measure similarity between the diffused image and the target caption using CLIP Score
https://arxiv.org/pdf/2104.08718.pdf

"""
import pathlib

import numpy as np
import torch
from PIL import Image
from torchmetrics.multimodal import CLIPScore
from tqdm import tqdm


class CLIPScoreCalculator:
    """CLIP score calculation for a dsd dataset

    uses torchmetrics implemnetation https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_score.html
    """

    def __init__(self, clip_model: str = "openai/clip-vit-base-patch16"):
        self._metric = CLIPScore(clip_model)

    @torch.no_grad()
    def calculate_scores_for_render_dir(self, render_dir: pathlib.Path):
        clip_scores_dict = {}
        # get subdirs
        subdirs = [d for d in render_dir.iterdir() if d.is_dir()]
        subdirs = [d for d in subdirs if d.name != "original"]
        for i, subdir in enumerate(subdirs):
            images = list(subdir.glob("*.png"))

            for image_path in images:
                image = np.array(Image.open(image_path))
                # check if image is all black -> NSFW filter -> skip
                if np.max(image) < 1e-4:
                    print(f"skipping {image_path} because it is all black")
                    continue

                image = torch.tensor(image).permute(2, 0, 1)
                caption = str(image_path.stem).split("/")[-1].split("_")[0]
                score = self._metric([image], [caption])
                # print(f"score for {image_path} with caption {caption} is  {score.item()}")
                clip_scores_dict[str(image_path)] = score.item()
        return clip_scores_dict

    def calculate_scores(self, render_path: pathlib.Path):
        render_paths = list(render_path.glob("**/rgb.png"))
        render_dirs = [p.parents[1] for p in render_paths]

        similarity_dict = {}
        for render_dir in tqdm(render_dirs):
            render_dir_simialrity_dict = self.calculate_scores_for_render_dir(render_dir)
            similarity_dict.update(render_dir_simialrity_dict)
        return similarity_dict


if __name__ == "__main__":
    from dsd import DATA_DIR

    render_dataset = DATA_DIR / "diffusion_renders" / "mugs" / "run_3"

    clip_scorer = CLIPScoreCalculator()
    similarity_dict = clip_scorer.calculate_scores(render_dataset)

    # set the dict keys to relative paths
    similarity_dict = {str(pathlib.Path(k).relative_to(render_dataset)): v for k, v in similarity_dict.items()}

    # dict to csv
    import pandas as pd

    df = pd.DataFrame.from_dict(similarity_dict, orient="index", columns=["clip_score"])
    # convert path to proper column and add new index
    df = df.reset_index()
    df = df.rename(columns={"index": "relative_path"})
    df.to_csv(str(render_dataset / "clip_scores.csv"))
