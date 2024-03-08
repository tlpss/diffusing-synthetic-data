import random

import click
from prompts import LIGHTINGS, STYLE_MEDIUM, mug_descriptions, table_descriptions

from dsd import DATA_DIR
from dsd.diffusion_rendering import (
    ControlNetFromDepthRenderer,
    ControlNetTXTFromDepthRenderer,
    SD2FromDepthRenderer,
    SD2InpaintingRenderer,
    SD15RealisticCheckpointControlNetFromDepthRenderer,
    SD15RealisticCheckpointControlNetTXTFromDepthRenderer,
    SDXLControlNetTXTFromDepthRenderer,
)
from dsd.generate_diffusion_renders import generate_diffusion_renders

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


source_directory = DATA_DIR / "renders" / "mugs" / "objaverse-filtered-2000"
target_directory = DATA_DIR / "diffusion_renders" / "mugs" / "cvpr" / "model-comparison-1-stage"

# clear if exists
# import shutil
# if target_directory.exists():
#     shutil.rmtree(target_directory)
target_directory.mkdir(parents=True, exist_ok=True)


num_images_per_prompt = 2
num_prompts_per_scene = 2
all_diffusion_renderers = [
    (SD15RealisticCheckpointControlNetFromDepthRenderer, {"num_images_per_prompt": num_images_per_prompt}),
    (SD15RealisticCheckpointControlNetTXTFromDepthRenderer, {"num_images_per_prompt": num_images_per_prompt}),
    (SD2FromDepthRenderer, {"num_images_per_prompt": num_images_per_prompt}),
    (SD2InpaintingRenderer, {"num_images_per_prompt": num_images_per_prompt}),
    (SDXLControlNetTXTFromDepthRenderer, {"num_images_per_prompt": num_images_per_prompt}),
    (ControlNetFromDepthRenderer, {"num_images_per_prompt": num_images_per_prompt}),
    (ControlNetTXTFromDepthRenderer, {"num_images_per_prompt": num_images_per_prompt}),
]


@click.command()
@click.option("--renderer_idx", type=int, default=-1)
@click.option("--create_coco_datasets", type=bool, default=False)
def main(renderer_idx, create_coco_datasets):
    if renderer_idx >= 0:
        diffusion_renderers = [all_diffusion_renderers[renderer_idx]]
    else:
        diffusion_renderers = all_diffusion_renderers
    generate_diffusion_renders(
        source_directory, target_directory, diffusion_renderers, prompts, num_prompts_per_scene=num_prompts_per_scene
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
