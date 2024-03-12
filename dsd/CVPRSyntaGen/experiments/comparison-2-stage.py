import random

import click
from prompts import LIGHTINGS, STYLE_MEDIUM, mug_descriptions, table_descriptions

from dsd import DATA_DIR
from dsd.cropped_diffusion_rendering import CropAndInpaintRenderer, CroppedRenderer
from dsd.diffusion_rendering import ControlNetFromDepthRenderer, SD2InpaintingRenderer, SD15InpaintingRenderer
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
target_directory = DATA_DIR / "diffusion_renders" / "mugs" / "cvpr" / "stage-2-comparison"

# clear if exists
# import shutil
# if target_directory.exists():
#     shutil.rmtree(target_directory)
target_directory.mkdir(parents=True, exist_ok=True)


num_images_per_prompt = 2
num_prompts_per_scene = 1

config_dict = {
    "renderer": {"class": ControlNetFromDepthRenderer, "args": {"num_images_per_prompt": num_images_per_prompt}},
    "cropped_renderer": {"class": CroppedRenderer, "args": {"bbox_padding": 10, "only_change_mask": True}},
    "inpainter": {"class": SD2InpaintingRenderer, "args": {"num_images_per_prompt": 1, "strength": 1}},
    "crop_and_inpainter": {"class": CropAndInpaintRenderer, "args": {"mask_dilation_iterations": 2}},
}


def build_model(config_dict):
    renderer = config_dict["renderer"]["class"](**config_dict["renderer"]["args"])
    cropped_renderer = config_dict["cropped_renderer"]["class"](
        renderer=renderer, **config_dict["cropped_renderer"]["args"]
    )
    inpainter = config_dict["inpainter"]["class"](**config_dict["inpainter"]["args"])
    crop_and_inpainter = config_dict["crop_and_inpainter"]["class"](
        crop_renderer=cropped_renderer, inpainter=inpainter, **config_dict["crop_and_inpainter"]["args"]
    )
    return crop_and_inpainter


# TODO: think of more elegant config generation
# Hydra?
all_diffusion_renderer_configs = [
    # no dilation on mask
    {
        "renderer": {"class": ControlNetFromDepthRenderer, "args": {"num_images_per_prompt": num_images_per_prompt}},
        "cropped_renderer": {"class": CroppedRenderer, "args": {"bbox_padding": 10, "only_change_mask": True}},
        "inpainter": {"class": SD2InpaintingRenderer, "args": {"num_images_per_prompt": 1, "strength": 1}},
        "crop_and_inpainter": {"class": CropAndInpaintRenderer, "args": {"mask_dilation_iterations": 0}},
    },
    # default
    {
        "renderer": {"class": ControlNetFromDepthRenderer, "args": {"num_images_per_prompt": num_images_per_prompt}},
        "cropped_renderer": {"class": CroppedRenderer, "args": {"bbox_padding": 10, "only_change_mask": True}},
        "inpainter": {"class": SD2InpaintingRenderer, "args": {"num_images_per_prompt": 1, "strength": 1}},
        "crop_and_inpainter": {"class": CropAndInpaintRenderer, "args": {"mask_dilation_iterations": 1}},
    },
    # larger dilation
    {
        "renderer": {"class": ControlNetFromDepthRenderer, "args": {"num_images_per_prompt": num_images_per_prompt}},
        "cropped_renderer": {"class": CroppedRenderer, "args": {"bbox_padding": 10, "only_change_mask": True}},
        "inpainter": {"class": SD2InpaintingRenderer, "args": {"num_images_per_prompt": 1, "strength": 1}},
        "crop_and_inpainter": {"class": CropAndInpaintRenderer, "args": {"mask_dilation_iterations": 2}},
    },
    # large dilation
    {
        "renderer": {"class": ControlNetFromDepthRenderer, "args": {"num_images_per_prompt": num_images_per_prompt}},
        "cropped_renderer": {"class": CroppedRenderer, "args": {"bbox_padding": 10, "only_change_mask": True}},
        "inpainter": {"class": SD2InpaintingRenderer, "args": {"num_images_per_prompt": 1, "strength": 1}},
        "crop_and_inpainter": {"class": CropAndInpaintRenderer, "args": {"mask_dilation_iterations": 5}},
    },
    # use bbox as starting point
    {
        "renderer": {"class": ControlNetFromDepthRenderer, "args": {"num_images_per_prompt": num_images_per_prompt}},
        "cropped_renderer": {"class": CroppedRenderer, "args": {"bbox_padding": 15, "only_change_mask": False}},
        "inpainter": {"class": SD2InpaintingRenderer, "args": {"num_images_per_prompt": 1, "strength": 1}},
        "crop_and_inpainter": {"class": CropAndInpaintRenderer, "args": {"mask_dilation_iterations": 3}},
    },
    # lower inpainting strength
    {
        "renderer": {"class": ControlNetFromDepthRenderer, "args": {"num_images_per_prompt": num_images_per_prompt}},
        "cropped_renderer": {"class": CroppedRenderer, "args": {"bbox_padding": 10, "only_change_mask": True}},
        "inpainter": {"class": SD2InpaintingRenderer, "args": {"num_images_per_prompt": 1, "strength": 0.8}},
        "crop_and_inpainter": {"class": CropAndInpaintRenderer, "args": {"mask_dilation_iterations": 1}},
    },
    # SD1.5 inpainting model
    {
        "renderer": {"class": ControlNetFromDepthRenderer, "args": {"num_images_per_prompt": num_images_per_prompt}},
        "cropped_renderer": {"class": CroppedRenderer, "args": {"bbox_padding": 10, "only_change_mask": True}},
        "inpainter": {"class": SD15InpaintingRenderer, "args": {"num_images_per_prompt": 1, "strength": 1}},
        "crop_and_inpainter": {"class": CropAndInpaintRenderer, "args": {"mask_dilation_iterations": 1}},
    },
]


class LazyInitialisationModels:
    "avoid memory overhead by initialising models only when needed"
    # TODO: is there a more elegant way to do this?
    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < len(all_diffusion_renderer_configs):
            self.idx += 1
            return build_model(all_diffusion_renderer_configs[self.idx - 1])
        else:
            raise StopIteration


all_diffusion_renderers = LazyInitialisationModels()


@click.command()
@click.option("--renderers_idx", type=int, multiple=True, default=[])
@click.option("--create_coco_datasets", type=bool, default=False)
def main(renderers_idx, create_coco_datasets):
    if len(renderers_idx) > 0:
        diffusion_renderers = [all_diffusion_renderers[i] for i in renderers_idx]
    else:
        diffusion_renderers = all_diffusion_renderers
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
