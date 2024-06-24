from paths import MUG_NO_TABLE_SCENES_DIR, SHOE_NO_TABLE_SCENES_DIR, TSHIRT_NO_TABLE_SCENES_DIR

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


diffusion_renderer = (ControlNetTXTFromDepthRenderer, {"num_images_per_prompt": 1})


def generate_renders(category):

    if category == "shoes":
        generate_diffusion_renders(
            SHOE_NO_TABLE_SCENES_DIR,
            DATA_DIR / "diffusion_renders" / "shoes" / "06-no-table-scenes",
            [diffusion_renderer],
            shoe_prompts,
            2,
        )
        generate_coco_datasets(
            DATA_DIR / "coco" / "shoes" / "diffusion_renders" / "06-no-table-scenes",
            DATA_DIR / "diffusion_renders" / "shoes" / "06-no-table-scenes",
            shoe_category,
        )

    elif category == "mugs":
        generate_diffusion_renders(
            MUG_NO_TABLE_SCENES_DIR,
            DATA_DIR / "diffusion_renders" / "mugs" / "06-no-table-scenes",
            [diffusion_renderer],
            mug_prompts,
            2,
        )
        generate_coco_datasets(
            DATA_DIR / "coco" / "mugs" / "diffusion_renders" / "06-no-table-scenes",
            DATA_DIR / "diffusion_renders" / "mugs" / "06-no-table-scenes",
            mug_category,
        )

    elif category == "tshirts":
        generate_diffusion_renders(
            TSHIRT_NO_TABLE_SCENES_DIR,
            DATA_DIR / "diffusion_renders" / "tshirts" / "06-no-table-scenes",
            [diffusion_renderer],
            tshirt_prompts,
            2,
        )
        generate_coco_datasets(
            DATA_DIR / "coco" / "tshirts" / "diffusion_renders" / "06-no-table-scenes",
            DATA_DIR / "diffusion_renders" / "tshirts" / "06-no-table-scenes",
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
