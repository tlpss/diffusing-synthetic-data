from paths import MUG_SCENES_DIR, SHOE_SCENES_DIR, TSHIRT_SCENES_DIR

from dsd import DATA_DIR
from dsd.cropped_diffusion_rendering import CropAndInpaintRenderer, CroppedRenderer
from dsd.diffusion_rendering import ControlNetTXTFromDepthRenderer, SD2InpaintingRenderer
from dsd.generate_coco_datasets_from_diffusion_renders import (
    generate_coco_datasets,
    mug_category,
    shoe_category,
    tshirt_category,
)
from dsd.generate_diffusion_renders import generate_crop_inpaint_diffusion_renders

prompt_dir = DATA_DIR.parent / "dsd" / "experiments" / "gemini_prompts"


mug_prompts = open(str(prompt_dir / "mug_prompts.txt"), "r").readlines()
mug_prompts = [(x.split(";")[0], x.split(";")[1].strip()) for x in mug_prompts]
mug_prompts, mug_background_prompts = [x[0] for x in mug_prompts], [x[1] for x in mug_prompts]

shoe_prompts = open(str(prompt_dir / "shoe_prompts.txt"), "r").readlines()
shoe_prompts = [(x.split(";")[0], x.split(";")[1].strip()) for x in shoe_prompts]
shoe_prompts, shoe_background_prompts = [x[0] for x in shoe_prompts], [x[1] for x in shoe_prompts]

tshirt_prompts = open(str(prompt_dir / "tshirt_prompts.txt"), "r").readlines()
tshirt_prompts = [(x.split(";")[0], x.split(";")[1].strip()) for x in tshirt_prompts]
tshirt_prompts, tshirt_background_prompts = [x[0] for x in tshirt_prompts], [x[1] for x in tshirt_prompts]


# TODO: think of more elegant config generation
# Hydra?
all_diffusion_renderer_configs = [
    # default
    {
        "renderer": {"class": ControlNetTXTFromDepthRenderer, "args": {"num_images_per_prompt": 1}},
        "cropped_renderer": {"class": CroppedRenderer, "args": {"bbox_padding": 10, "only_change_mask": True}},
        "inpainter": {"class": SD2InpaintingRenderer, "args": {"num_images_per_prompt": 1, "strength": 1}},
        "crop_and_inpainter": {"class": CropAndInpaintRenderer, "args": {"mask_dilation_iterations": 3}},
    },
]


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


class LazyInitialisationModels:
    "avoid memory overhead by initialising models only when needed"
    # TODO: is there a more elegant way to do this?
    def __init__(self, configs):
        self.configs = configs

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < len(self.configs):
            self.idx += 1
            return build_model(self.configs[self.idx - 1])
        else:
            raise StopIteration


diffusion_renderers = LazyInitialisationModels(all_diffusion_renderer_configs)


def generate_renders(category):

    if category == "shoes":
        generate_crop_inpaint_diffusion_renders(
            SHOE_SCENES_DIR,
            DATA_DIR / "diffusion_renders" / "shoes" / "04a-two-stage-larger-mask",
            diffusion_renderers,
            shoe_prompts,
            shoe_background_prompts,
            2,
        )
        generate_coco_datasets(
            DATA_DIR / "coco" / "shoes" / "diffusion_renders" / "04a-two-stage-larger-mask",
            DATA_DIR / "diffusion_renders" / "shoes" / "04a-two-stage-larger-mask",
            shoe_category,
        )

    elif category == "mugs":
        generate_crop_inpaint_diffusion_renders(
            MUG_SCENES_DIR,
            DATA_DIR / "diffusion_renders" / "mugs" / "04a-two-stage-larger-mask",
            diffusion_renderers,
            mug_prompts,
            mug_background_prompts,
            2,
        )
        generate_coco_datasets(
            DATA_DIR / "coco" / "mugs" / "diffusion_renders" / "04a-two-stage-larger-mask",
            DATA_DIR / "diffusion_renders" / "mugs" / "04a-two-stage-larger-mask",
            mug_category,
        )

    elif category == "tshirts":
        generate_crop_inpaint_diffusion_renders(
            TSHIRT_SCENES_DIR,
            DATA_DIR / "diffusion_renders" / "tshirts" / "04a-two-stage-larger-mask",
            diffusion_renderers,
            tshirt_prompts,
            tshirt_background_prompts,
            2,
        )
        generate_coco_datasets(
            DATA_DIR / "coco" / "tshirts" / "diffusion_renders" / "04a-two-stage-larger-mask",
            DATA_DIR / "diffusion_renders" / "tshirts" / "04a-two-stage-larger-mask",
            tshirt_category,
        )

    else:
        print("Invalid category")


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--categories", type=str, multiple=True, default=["shoes", "mugs", "tshirts"])
    def generate_renders_cli(categories):
        for category in categories:
            generate_renders(category)

    generate_renders_cli()
