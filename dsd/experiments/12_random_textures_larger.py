# add paths.py to the python path
import sys

from dsd import DATA_DIR
from dsd.render_random_textures import generate_random_texture_renders

sys.path.append(str(DATA_DIR.parent / "dsd" / "experiments"))

from paths import MUG_SCENES_DIR, RANDOM_TEXTURE_BASELINE_DIR, SHOE_SCENES_DIR, TSHIRT_SCENES_DIR

num_renders_per_scene = 4


def generate_w_table():
    generate_random_texture_renders(
        source_directory=SHOE_SCENES_DIR,
        target_directory=RANDOM_TEXTURE_BASELINE_DIR / "shoes" / "large-run",
        num_renders_per_scene=num_renders_per_scene,
    )

    generate_random_texture_renders(
        source_directory=MUG_SCENES_DIR,
        target_directory=RANDOM_TEXTURE_BASELINE_DIR / "mugs" / "large-run",
        num_renders_per_scene=num_renders_per_scene,
    )

    generate_random_texture_renders(
        source_directory=TSHIRT_SCENES_DIR,
        target_directory=RANDOM_TEXTURE_BASELINE_DIR / "tshirts" / "large-run",
        num_renders_per_scene=num_renders_per_scene,
    )


if __name__ == "__main__":
    generate_w_table()
    from paths import (  # noqa
        RANDOM_TEXTURE_MUG_LARGE_DATASET,
        RANDOM_TEXTURE_SHOE_LARGE_DATASET,
        RANDOM_TEXTURE_TSHIRT_LARGE_DATASET,
    )

    from dsd.generate_coco_datasets_from_diffusion_renders import (
        generate_coco_datasets,
        mug_category,
        shoe_category,
        tshirt_category,
    )

    generate_coco_datasets(
        target_coco_path=RANDOM_TEXTURE_SHOE_LARGE_DATASET.parent,
        render_path=RANDOM_TEXTURE_BASELINE_DIR / "shoes" / "large-run",
        coco_category=shoe_category,
    )

    generate_coco_datasets(
        target_coco_path=RANDOM_TEXTURE_MUG_LARGE_DATASET.parent,
        render_path=RANDOM_TEXTURE_BASELINE_DIR / "mugs" / "large-run",
        coco_category=mug_category,
    )

    generate_coco_datasets(
        target_coco_path=RANDOM_TEXTURE_TSHIRT_LARGE_DATASET.parent,
        render_path=RANDOM_TEXTURE_BASELINE_DIR / "tshirts" / "large-run",
        coco_category=tshirt_category,
    )
