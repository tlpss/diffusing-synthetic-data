from paths import MUG_SCENES_DIR, SHOE_SCENES_DIR, TSHIRT_SCENES_DIR

from dsd import DATA_DIR
from dsd.cropped_diffusion_rendering import DualInpaintRenderer
from dsd.diffusion_rendering import SD2InpaintingRenderer
from dsd.generate_coco_datasets_from_diffusion_renders import (
    generate_coco_datasets,
    mug_category,
    shoe_category,
    tshirt_category,
)
from dsd.generate_diffusion_renders import generate_dual_inpaint_renders

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
inpainter = SD2InpaintingRenderer(1)
dual_inpainter = DualInpaintRenderer(inpainter=inpainter, mask_dilation_iterations=1)
diffusion_renderers = [dual_inpainter]


def generate_renders(category):

    if category == "shoes":
        generate_dual_inpaint_renders(
            SHOE_SCENES_DIR,
            DATA_DIR / "diffusion_renders" / "shoes" / "08-dual-inpainting",
            diffusion_renderers,
            shoe_prompts,
            shoe_background_prompts,
            2,
        )
        generate_coco_datasets(
            DATA_DIR / "coco" / "shoes" / "diffusion_renders" / "08-dual-inpainting",
            DATA_DIR / "diffusion_renders" / "shoes" / "08-dual-inpainting",
            shoe_category,
        )

    elif category == "mugs":
        generate_dual_inpaint_renders(
            MUG_SCENES_DIR,
            DATA_DIR / "diffusion_renders" / "mugs" / "08-dual-inpainting",
            diffusion_renderers,
            mug_prompts,
            mug_background_prompts,
            2,
        )
        generate_coco_datasets(
            DATA_DIR / "coco" / "mugs" / "diffusion_renders" / "08-dual-inpainting",
            DATA_DIR / "diffusion_renders" / "mugs" / "08-dual-inpainting",
            mug_category,
        )

    elif category == "tshirts":
        generate_dual_inpaint_renders(
            TSHIRT_SCENES_DIR,
            DATA_DIR / "diffusion_renders" / "tshirts" / "08-dual-inpainting",
            diffusion_renderers,
            tshirt_prompts,
            tshirt_background_prompts,
            2,
        )
        generate_coco_datasets(
            DATA_DIR / "coco" / "tshirts" / "diffusion_renders" / "08-dual-inpainting",
            DATA_DIR / "diffusion_renders" / "tshirts" / "08-dual-inpainting",
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
