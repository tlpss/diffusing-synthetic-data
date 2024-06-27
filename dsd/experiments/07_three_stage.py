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
from dsd.generate_diffusion_renders import generate_three_stage_renders
from dsd.three_stage_diffusion_rendering import ThreeStageRenderer

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


cropped_renderer = ControlNetTXTFromDepthRenderer(num_images_per_prompt=1)
cropped_renderer = CroppedRenderer(cropped_renderer)
inpaint_renderer = SD2InpaintingRenderer(num_images_per_prompt=1)
crop_and_inpain_renderer = CropAndInpaintRenderer(cropped_renderer, inpaint_renderer, mask_dilation_iterations=1)
three_stage_renderer = ThreeStageRenderer(crop_and_inpain_renderer, cropped_renderer)


def generate_renders(category):

    if category == "shoes":
        generate_three_stage_renders(
            SHOE_SCENES_DIR,
            DATA_DIR / "diffusion_renders" / "shoes" / "07-three-stage",
            [three_stage_renderer],
            shoe_prompts,
            shoe_background_prompts,
            2,
        )
        generate_coco_datasets(
            DATA_DIR / "coco" / "shoes" / "diffusion_renders" / "07-three-stage",
            DATA_DIR / "diffusion_renders" / "shoes" / "07-three-stage",
            shoe_category,
        )

    elif category == "mugs":
        pass

        generate_three_stage_renders(
            MUG_SCENES_DIR,
            DATA_DIR / "diffusion_renders" / "mugs" / "07-three-stage",
            [three_stage_renderer],
            mug_prompts,
            mug_background_prompts,
            2,
        )
        generate_coco_datasets(
            DATA_DIR / "coco" / "mugs" / "diffusion_renders" / "07-three-stage",
            DATA_DIR / "diffusion_renders" / "mugs" / "07-three-stage",
            mug_category,
        )

    elif category == "tshirts":
        generate_three_stage_renders(
            TSHIRT_SCENES_DIR,
            DATA_DIR / "diffusion_renders" / "tshirts" / "07-three-stage",
            [three_stage_renderer],
            tshirt_prompts,
            tshirt_background_prompts,
            2,
        )
        generate_coco_datasets(
            DATA_DIR / "coco" / "tshirts" / "diffusion_renders" / "07-three-stage",
            DATA_DIR / "diffusion_renders" / "tshirts" / "07-three-stage",
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
