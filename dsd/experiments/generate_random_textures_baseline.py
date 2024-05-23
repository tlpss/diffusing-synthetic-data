# add paths.py to the python path
import sys

from dsd import DATA_DIR
from dsd.render_random_textures import generate_random_texture_renders

sys.path.append(str(DATA_DIR.parent / "dsd" / "experiments"))

print(sys.path)
import click
from paths import (
    MUG_RANDOM_TEXTURE_RENDER_DIR,
    MUG_SCENES_DIR,
    SHOE_RANDOM_TEXTURE_RENDER_DIR,
    SHOE_SCENES_DIR,
    TSHIRT_RANDOM_TEXTURE_RENDER_DIR,
    TSHIRT_SCENES_DIR,
)


@click.command()
@click.option("--num_renders_per_scene", default=1, help="Number of renders per scene")
@click.option("--category", "-c", multiple=True, help="Category of the object to render random textures for")
def main(num_renders_per_scene, category):
    # convert tuple to list
    category = list(category)
    breakpoint()
    if "shoes" in category:
        category.remove("shoes")
        generate_random_texture_renders(
            source_directory=SHOE_SCENES_DIR,
            target_directory=SHOE_RANDOM_TEXTURE_RENDER_DIR,
            num_renders_per_scene=num_renders_per_scene,
        )
    if "mugs" in category:
        category.remove("mugs")
        generate_random_texture_renders(
            source_directory=MUG_SCENES_DIR,
            target_directory=MUG_RANDOM_TEXTURE_RENDER_DIR,
            num_renders_per_scene=num_renders_per_scene,
        )
    if "tshirts" in category:
        category.remove("tshirts")
        generate_random_texture_renders(
            source_directory=TSHIRT_SCENES_DIR,
            target_directory=TSHIRT_RANDOM_TEXTURE_RENDER_DIR,
            num_renders_per_scene=num_renders_per_scene,
        )

    if len(category) > 0:
        raise ValueError(f"Category {category} not recognized. Please choose from 'shoes', 'mugs', 'tshirts'.")


if __name__ == "__main__":
    main()
