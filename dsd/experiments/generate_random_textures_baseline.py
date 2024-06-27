# add paths.py to the python path
import sys

from dsd import DATA_DIR
from dsd.render_random_textures import generate_random_texture_renders

sys.path.append(str(DATA_DIR.parent / "dsd" / "experiments"))

from paths import (
    MUG_RANDOM_TEXTURE_NO_TABLE_RENDER_DIR,
    MUG_RANDOM_TEXTURE_RENDER_DIR,
    MUG_SCENES_DIR,
    SHOE_RANDOM_TEXTURE_NO_TABLE_RENDER_DIR,
    SHOE_RANDOM_TEXTURE_RENDER_DIR,
    SHOE_SCENES_DIR,
    TSHIRT_NO_TABLE_SCENES_DIR,
    TSHIRT_RANDOM_TEXTURE_NO_TABLE_RENDER_DIR,
    TSHIRT_RANDOM_TEXTURE_RENDER_DIR,
    TSHIRT_SCENES_DIR,
)

num_renders_per_scene = 2


def generate_w_table():
    generate_random_texture_renders(
        source_directory=SHOE_SCENES_DIR,
        target_directory=SHOE_RANDOM_TEXTURE_RENDER_DIR,
        num_renders_per_scene=num_renders_per_scene,
    )

    generate_random_texture_renders(
        source_directory=MUG_SCENES_DIR,
        target_directory=MUG_RANDOM_TEXTURE_RENDER_DIR,
        num_renders_per_scene=num_renders_per_scene,
    )

    generate_random_texture_renders(
        source_directory=TSHIRT_SCENES_DIR,
        target_directory=TSHIRT_RANDOM_TEXTURE_RENDER_DIR,
        num_renders_per_scene=num_renders_per_scene,
    )


def generate_wo_table():
    from dsd.render_random_textures import Config

    config = Config(scene_contains_table=False)
    generate_random_texture_renders(
        source_directory=TSHIRT_NO_TABLE_SCENES_DIR,
        target_directory=TSHIRT_RANDOM_TEXTURE_NO_TABLE_RENDER_DIR,
        num_renders_per_scene=num_renders_per_scene,
        config=config,
    )

    # generate_random_texture_renders(
    #     source_directory=MUG_NO_TABLE_SCENES_DIR,
    #     target_directory=MUG_RANDOM_TEXTURE_NO_TABLE_RENDER_DIR,
    #     num_renders_per_scene=num_renders_per_scene,
    #     config=config
    # )

    # generate_random_texture_renders(
    #     source_directory=SHOE_NO_TABLE_SCENES_DIR,
    #     target_directory=SHOE_RANDOM_TEXTURE_NO_TABLE_RENDER_DIR,
    #     num_renders_per_scene=num_renders_per_scene,
    #     config=config
    # )


if __name__ == "__main__":
    # generate_w_table()
    # generate_wo_table()
    from paths import (  # noqa
        RANDOM_TEXTURE_BASELINE_MUG_DATASET,
        RANDOM_TEXTURE_BASELINE_MUG_NO_TABLE_DATASET,
        RANDOM_TEXTURE_BASELINE_SHOE_DATASET,
        RANDOM_TEXTURE_BASELINE_SHOE_NO_TABLE_DATASET,
        RANDOM_TEXTURE_BASELINE_TSHIRT_DATASET,
        RANDOM_TEXTURE_BASELINE_TSHIRT_NO_TABLE_DATASET,
    )

    from dsd.generate_coco_datasets_from_diffusion_renders import (
        generate_coco_datasets,
        mug_category,
        shoe_category,
        tshirt_category,
    )

    # generate_coco_datasets(
    #     target_coco_path=RANDOM_TEXTURE_BASELINE_SHOE_DATASET,
    #     render_path=SHOE_RANDOM_TEXTURE_RENDER_DIR,
    #     coco_category="shoe",
    # )
    # generate_coco_datasets(
    #     target_coco_path=RANDOM_TEXTURE_BASELINE_MUG_DATASET,
    #     render_path=MUG_RANDOM_TEXTURE_RENDER_DIR,
    #     coco_category="mug",
    # )
    # generate_coco_datasets(
    #     target_coco_path=RANDOM_TEXTURE_BASELINE_TSHIRT_DATASET,
    #     render_path=TSHIRT_RANDOM_TEXTURE_RENDER_DIR,
    #     coco_category="tshirt",
    # )

    generate_coco_datasets(
        target_coco_path=RANDOM_TEXTURE_BASELINE_SHOE_NO_TABLE_DATASET.parent,
        render_path=SHOE_RANDOM_TEXTURE_NO_TABLE_RENDER_DIR,
        coco_category=shoe_category,
    )

    generate_coco_datasets(
        target_coco_path=RANDOM_TEXTURE_BASELINE_MUG_NO_TABLE_DATASET.parent,
        render_path=MUG_RANDOM_TEXTURE_NO_TABLE_RENDER_DIR,
        coco_category=mug_category,
    )

    generate_coco_datasets(
        target_coco_path=RANDOM_TEXTURE_BASELINE_TSHIRT_NO_TABLE_DATASET.parent,
        render_path=TSHIRT_RANDOM_TEXTURE_NO_TABLE_RENDER_DIR,
        coco_category=tshirt_category,
    )
