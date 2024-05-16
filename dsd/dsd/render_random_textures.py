# load a directory with blend files

# open each blend file
# for i in range N
# add random polyhaven material to the object and the table, add random HDRI to the world?
# render the scene
# save the rendered image
import pathlib
import random
import shutil

import bpy
import numpy as np
import tqdm

from dsd.rendering.blender.polyhaven.polyhaven_backgrounds import (
    PolyhavenHDRIConfig,
    add_polyhaven_hdri_background_to_scene,
)
from dsd.rendering.blender.polyhaven.polyhaven_materials import add_texture_to_object
from dsd.rendering.blender.renderer import CyclesRendererConfig, render_scene


def generate_random_texture_renders(source_directory, target_directory, num_renders_per_scene=1):
    blender_scene_paths = list(source_directory.glob("**/scene.blend"))
    image_dirs = [p.parent for p in blender_scene_paths]
    image_dirs = sorted(image_dirs)

    # fix seeds to make renders reproducible
    random.seed(2024)
    np.random.seed(2024)

    for image_dir in tqdm.tqdm(image_dirs):
        relative_path_to_source_dir = image_dir.relative_to(source_directory)
        image_target_dir = target_directory / relative_path_to_source_dir
        image_target_dir.mkdir(parents=True, exist_ok=True)
        # copy the orignal images
        blender_image_target_dir = image_target_dir / "original"
        blender_image_target_dir.mkdir(parents=True, exist_ok=True)
        for image_path in image_dir.glob("*"):
            shutil.copy(image_path, blender_image_target_dir)

        for i in range(num_renders_per_scene):
            bpy.ops.wm.open_mainfile(filepath=str(image_dir / "scene.blend"))

            # get the cube object and select it
            cube = bpy.data.objects["Cube"]

            # get the 'object'
            mesh_object = bpy.data.objects["object"]

            # add a random polyhaven material to the cube
            add_texture_to_object(cube)
            add_texture_to_object(mesh_object)

            # add a random polyhaven HDRI background to the scene
            add_polyhaven_hdri_background_to_scene(PolyhavenHDRIConfig())

            # render the scene
            render_scene(
                CyclesRendererConfig(render_depth=False, render_segmentation=False, render_rgb=True),
                str(image_target_dir / f"render_{i}"),
            )


if __name__ == "__main__":

    generate_random_texture_renders(
        source_directory=pathlib.Path(
            "/home/tlips/Code/diffusing-synthetic-data/data/renders/shoes/2024-05-14_12-17-54"
        ),
        target_directory=pathlib.Path("/home/tlips/Code/diffusing-synthetic-data/data/textures/shoes/test"),
        num_renders_per_scene=1,
    )
