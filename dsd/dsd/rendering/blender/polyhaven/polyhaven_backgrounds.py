""" random HDRI backgrounds in python
 code copied from https://github.com/tlpss/synthetic-cloth-data/blob/main/synthetic-cloth-data/synthetic_cloth_data/synthetic_images/scene_builder/background.py"""

import dataclasses
import json

from dsd.rendering.blender.polyhaven import POLYHAVEN_ASSETS_SNAPSHOT_PATH


@dataclasses.dataclass
class PolyhavenHDRIConfig:
    assets_json_path: str = POLYHAVEN_ASSETS_SNAPSHOT_PATH

    asset_list = json.load(open(assets_json_path, "r"))["assets"]
    asset_list = [asset for asset in asset_list if asset["type"] == "worlds"]


import dataclasses

import airo_blender as ab
import bpy
import numpy as np


def add_polyhaven_hdri_background_to_scene(config: PolyhavenHDRIConfig):
    """adds a polyhaven HDRI background to the scene."""
    hdri_dict = np.random.choice(config.asset_list)
    world = ab.load_asset(**hdri_dict)
    bpy.context.scene.world = world
    breakpoint()
    # set Polyhaven HDRI resolution to 4k
    # requires creating manual context override, although this is not documented ofc.
    # override = bpy.context.copy()
    # override["world"] = bpy.context.scene.world
    # with bpy.context.temp_override(**override):
    #     bpy.ops.pha.resolution_switch(res="4k", asset_id=bpy.context.world.name)
    return world
