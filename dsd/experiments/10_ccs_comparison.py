from paths import MUG_SCENES_DIR, SHOE_SCENES_DIR, TSHIRT_SCENES_DIR

from dsd import DATA_DIR
from dsd.diffusion_rendering import ControlNetTXTFromDepthRenderer
from dsd.generate_coco_datasets_from_diffusion_renders import (
    generate_coco_datasets,
    mug_category,
    shoe_category,
    tshirt_category,
)
from dsd.generate_diffusion_renders import generate_diffusion_renders

prompt_dir = DATA_DIR.parent / "dsd" / "experiments" / "gemini_prompts"


mug_prompts = open(str(prompt_dir / "mug_prompts.txt"), "r").readlines()
mug_prompts = [(x.split(";")[0], x.split(";")[1].strip()) for x in mug_prompts]
mug_prompts = [f"{x[0]} on a {x[1]}" for x in mug_prompts]

shoe_prompts = open(str(prompt_dir / "shoe_prompts.txt"), "r").readlines()
shoe_prompts = [(x.split(";")[0], x.split(";")[1].strip()) for x in shoe_prompts]
shoe_prompts = [f"{x[0]} on a {x[1]}" for x in shoe_prompts]

tshirt_prompts = open(str(prompt_dir / "tshirt_prompts.txt"), "r").readlines()
tshirt_prompts = [(x.split(";")[0], x.split(";")[1].strip()) for x in tshirt_prompts]
tshirt_prompts = [f"{x[0]} on a {x[1]}" for x in tshirt_prompts]


diffusion_renderers = [
    (ControlNetTXTFromDepthRenderer, {"num_images_per_prompt": 1, "controlnet_conditioning_scale": 0.5}),
    (ControlNetTXTFromDepthRenderer, {"num_images_per_prompt": 1, "controlnet_conditioning_scale": 1.0}),
    (ControlNetTXTFromDepthRenderer, {"num_images_per_prompt": 1, "controlnet_conditioning_scale": 1.5}),
    (ControlNetTXTFromDepthRenderer, {"num_images_per_prompt": 1, "controlnet_conditioning_scale": 2.0}),
    (ControlNetTXTFromDepthRenderer, {"num_images_per_prompt": 1, "controlnet_conditioning_scale": 2.5}),
]


def generate_renders(category):

    if category == "shoes":
        generate_diffusion_renders(
            SHOE_SCENES_DIR,
            DATA_DIR / "diffusion_renders" / "shoes" / "10-ccs-comparison",
            diffusion_renderers,
            shoe_prompts,
            2,
        )
        generate_coco_datasets(
            DATA_DIR / "coco" / "shoes" / "diffusion_renders" / "10-ccs-comparison",
            DATA_DIR / "diffusion_renders" / "shoes" / "10-ccs-comparison",
            shoe_category,
        )

    elif category == "mugs":
        generate_diffusion_renders(
            MUG_SCENES_DIR,
            DATA_DIR / "diffusion_renders" / "mugs" / "10-ccs-comparison",
            diffusion_renderers,
            mug_prompts,
            2,
        )
        generate_coco_datasets(
            DATA_DIR / "coco" / "mugs" / "diffusion_renders" / "10-ccs-comparison",
            DATA_DIR / "diffusion_renders" / "mugs" / "10-ccs-comparison",
            mug_category,
        )

    elif category == "tshirts":
        generate_diffusion_renders(
            TSHIRT_SCENES_DIR,
            DATA_DIR / "diffusion_renders" / "tshirts" / "10-ccs-comparison",
            diffusion_renderers,
            tshirt_prompts,
            2,
        )
        generate_coco_datasets(
            DATA_DIR / "coco" / "tshirts" / "diffusion_renders" / "10-ccs-comparison",
            DATA_DIR / "diffusion_renders" / "tshirts" / "10-ccs-comparison",
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
