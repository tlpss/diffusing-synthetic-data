# add paths.py to the python path
import sys

from dsd import DATA_DIR
from dsd.render_random_textures import generate_random_texture_renders

sys.path.append(str(DATA_DIR.parent / "dsd" / "experiments"))

from paths import (
    MUG_RANDOM_TEXTURE_RENDER_DIR,
    MUG_SCENES_DIR,
    SHOE_RANDOM_TEXTURE_RENDER_DIR,
    SHOE_SCENES_DIR,
    TSHIRT_RANDOM_TEXTURE_RENDER_DIR,
    TSHIRT_SCENES_DIR,
)

num_renders_per_scene = 2

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
