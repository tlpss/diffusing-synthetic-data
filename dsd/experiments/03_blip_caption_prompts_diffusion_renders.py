import json

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

prompt_dir = DATA_DIR.parent / "dsd" / "experiments" / "blip_captions"


def _convert_blip_captions_to_prompts(caption_file_path):
    with open(caption_file_path) as f:
        captions = json.load(f)
    prompts = []
    for caption_pair in captions.values():
        prompts.extend(caption_pair)

    # remove "/" from prompts
    prompts = [prompt.replace("/", " ") for prompt in prompts]

    print("removing / from prompts")
    return prompts


mug_prompts = _convert_blip_captions_to_prompts(prompt_dir / "mug-captions.json")
shoe_prompts = _convert_blip_captions_to_prompts(prompt_dir / "shoe-captions.json")
tshirt_prompts = _convert_blip_captions_to_prompts(prompt_dir / "tshirt-captions.json")


diffusion_renderer = (ControlNetTXTFromDepthRenderer, {"num_images_per_prompt": 1})


def generate_renders(category):

    if category == "shoes":
        generate_diffusion_renders(
            SHOE_SCENES_DIR,
            DATA_DIR / "diffusion_renders" / "shoes" / "03-blip-captions",
            [diffusion_renderer],
            shoe_prompts,
            2,
        )
        generate_coco_datasets(
            DATA_DIR / "coco" / "shoes" / "diffusion_renders" / "03-blip-captions",
            DATA_DIR / "diffusion_renders" / "shoes" / "03-blip-captions",
            shoe_category,
        )

    elif category == "mugs":
        generate_diffusion_renders(
            MUG_SCENES_DIR,
            DATA_DIR / "diffusion_renders" / "mugs" / "03-blip-captions",
            [diffusion_renderer],
            mug_prompts,
            2,
        )
        generate_coco_datasets(
            DATA_DIR / "coco" / "mugs" / "diffusion_renders" / "03-blip-captions",
            DATA_DIR / "diffusion_renders" / "mugs" / "03-blip-captions",
            mug_category,
        )

    elif category == "tshirts":
        generate_diffusion_renders(
            TSHIRT_SCENES_DIR,
            DATA_DIR / "diffusion_renders" / "tshirts" / "03-blip-captions",
            [diffusion_renderer],
            tshirt_prompts,
            2,
        )
        generate_coco_datasets(
            DATA_DIR / "coco" / "tshirts" / "diffusion_renders" / "03-blip-captions",
            DATA_DIR / "diffusion_renders" / "tshirts" / "03-blip-captions",
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
