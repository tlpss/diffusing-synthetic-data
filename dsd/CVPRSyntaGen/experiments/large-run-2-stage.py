import random

import click
from prompts import LIGHTINGS, STYLE_MEDIUM, mug_descriptions, table_descriptions

from dsd import DATA_DIR
from dsd.cropped_diffusion_rendering import CropAndInpaintRenderer, CroppedRenderer
from dsd.diffusion_rendering import ControlNetFromDepthRenderer, SD2InpaintingRenderer
from dsd.generate_diffusion_renders import generate_crop_inpaint_diffusion_renders

random.seed(2024)

prompts = []
for _ in range(1000):
    mug_desc = random.choice(mug_descriptions)
    table_desc = random.choice(table_descriptions)
    prompt = f"A {mug_desc} on a {table_desc}, "
    prompts.append(prompt)


prompts = [prompt + ", " + random.choice(LIGHTINGS) + ", " + random.choice(STYLE_MEDIUM) for prompt in prompts]
# remove double commas
prompts = [prompt.replace(", ,", ",") for prompt in prompts]
# remove trailing comma at the end of the string
prompts = [prompt.rstrip(", ") for prompt in prompts]


source_directory = DATA_DIR / "renders" / "mugs" / "objaverse-filtered-2500"
target_directory = DATA_DIR / "diffusion_renders" / "mugs" / "cvpr" / "large-run-2-stage"

# clear if exists
# import shutil
# if target_directory.exists():
#     shutil.rmtree(target_directory)
target_directory.mkdir(parents=True, exist_ok=True)


num_images_per_prompt = 2
num_prompts_per_scene = 2


renderer = ControlNetFromDepthRenderer(num_images_per_prompt=num_images_per_prompt)
cropped_renderer = CroppedRenderer(renderer=renderer, bbox_padding=15, only_change_mask=False)
inpainter = SD2InpaintingRenderer(num_images_per_prompt=1, strength=1)
crop_and_inpainter = CropAndInpaintRenderer(
    crop_renderer=cropped_renderer, inpainter=inpainter, mask_dilation_iterations=3
)


@click.command()
@click.option("--renderers_idx", type=int, multiple=True, default=[])
@click.option("--create_coco_datasets", type=bool, default=False)
def main(renderers_idx, create_coco_datasets):
    diffusion_renderers = [crop_and_inpainter]
    generate_crop_inpaint_diffusion_renders(
        source_directory,
        target_directory,
        diffusion_renderers,
        mug_descriptions,
        table_descriptions,
        num_prompts_per_scene=num_prompts_per_scene,
    )

    if create_coco_datasets:
        # generate COCO datasets
        from dsd.generate_coco_datasets_from_diffusion_renders import CocoKeypointCategory, generate_coco_datasets

        category = CocoKeypointCategory(
            id=0, name="mug", supercategory="mug", keypoints=["bottom", "handle", "top"], skeleton=[[0, 1], [1, 2]]
        )
        generate_coco_datasets(target_directory, category)


if __name__ == "__main__":
    main()
