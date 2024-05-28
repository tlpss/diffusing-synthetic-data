""" prompting strategie comparisions : class name prompt
"""
from paths import DATA_DIR, MUG_SCENES_DIR, SHOE_SCENES_DIR, TSHIRT_SCENES_DIR

from dsd.diffusion_rendering import ControlNetFromDepthRenderer
from dsd.generate_coco_datasets_from_diffusion_renders import (
    generate_coco_datasets,
    mug_category,
    shoe_category,
    tshirt_category,
)
from dsd.generate_diffusion_renders import generate_diffusion_renders

diffusion_renderer = (ControlNetFromDepthRenderer, {"num_images_per_prompt": 2})


def generate_renders(category):

    if category == "shoes":
        generate_diffusion_renders(
            SHOE_SCENES_DIR,
            DATA_DIR / "diffusion_renders" / "shoes" / "01-prompt-classname",
            [diffusion_renderer],
            [" A photo of a Shoe"],
            1,
        )
        generate_coco_datasets(
            DATA_DIR / "coco" / "shoes" / "diffusion_renders" / "01-prompt-classname",
            DATA_DIR / "diffusion_renders" / "shoes" / "01-prompt-classname",
            shoe_category,
        )

    elif category == "mugs":
        generate_diffusion_renders(
            MUG_SCENES_DIR,
            DATA_DIR / "diffusion_renders" / "mugs" / "01-prompt-classname",
            [diffusion_renderer],
            [" A photo of a Mug"],
            1,
        )
        generate_coco_datasets(
            DATA_DIR / "coco" / "mugs" / "diffusion_renders" / "01-prompt-classname",
            DATA_DIR / "diffusion_renders" / "mugs" / "01-prompt-classname",
            mug_category,
        )

    elif category == "tshirts":
        generate_diffusion_renders(
            TSHIRT_SCENES_DIR,
            DATA_DIR / "diffusion_renders" / "tshirts" / "01-prompt-classname",
            [diffusion_renderer],
            [" A photo of a Tshirt"],
            1,
        )
        generate_coco_datasets(
            DATA_DIR / "coco" / "tshirts" / "diffusion_renders" / "01-prompt-classname",
            DATA_DIR / "diffusion_renders" / "tshirts" / "01-prompt-classname",
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
