"""code to add random polyhaven textures to blender objects.
copied from https://github.com/tlpss/synthetic-cloth-data/tree/main
"""
from __future__ import annotations

import dataclasses
import json

import airo_blender as ab
import bpy
import cv2
import numpy as np

from dsd.rendering.blender.polyhaven import POLYHAVEN_ASSETS_SNAPSHOT_PATH


@dataclasses.dataclass
class PolyhavenMaterials:
    assets_json_path: str = POLYHAVEN_ASSETS_SNAPSHOT_PATH

    asset_list = json.load(open(assets_json_path, "r"))["assets"]
    asset_list = [asset for asset in asset_list if asset["type"] == "materials"]

    polyhaven_material_probability = 0.999


def sample_hsv_color():
    """hsv color with h,s,v in range (0,1) as in blender"""
    hue = np.random.uniform(0, 1.0)
    saturation = np.random.uniform(0.0, 1)
    value = np.random.uniform(0.0, 1)
    return np.array([hue, saturation, value])


def hsv_to_rgb(hsv: np.ndarray):
    """converts hsv in range (0,1) to rgb in range (0,1)"""
    assert hsv.shape == (3,)
    assert np.all(hsv <= 1.0), "hsv values must be in range (0,1)"
    hsv = hsv.astype(np.float32)
    hsv[0] *= 360  # convert from (0,1) to degrees as in blender
    rgb = cv2.cvtColor(hsv[np.newaxis, np.newaxis, ...], cv2.COLOR_HSV2RGB)
    return rgb[0][0]


def add_texture_to_object(object: bpy.types.Object) -> bpy.types.Object:
    if np.random.rand() < PolyhavenMaterials.polyhaven_material_probability and len(PolyhavenMaterials.asset_list) > 0:
        print("polyhaven material")
        material_dict = np.random.choice(PolyhavenMaterials.asset_list)
        material = ab.load_asset(**material_dict)
        assert isinstance(material, bpy.types.Material)

        # add a color mix node before the principled BSDF color
        # to randomize the base color hue

        # use multiply to limit the change in brightness (which is always an issue with addition)
        # colors should be close to (1,1,1) to avoid darkening the material too much (this is the issue with multiplying..)
        # so set value to 1 and keep saturation low.
        hue = np.random.uniform(0, 1)
        saturation = np.random.uniform(0.0, 0.7)
        value = 1.0
        base_hsv = np.array([hue, saturation, value])
        base_rgb = hsv_to_rgb(base_hsv)

        multiply_node = material.node_tree.nodes.new("ShaderNodeMixRGB")
        multiply_node.blend_type = "MULTIPLY"
        multiply_node.inputs["Fac"].default_value = 1.0
        multiply_node.inputs["Color2"].default_value = (*base_rgb, 1.0)

        # map original input of the BSDF base color to the multiply node
        # cannot search on "Name" because they can have suffixes like ".001"
        for node in material.node_tree.nodes:
            if isinstance(node, bpy.types.ShaderNodeBsdfPrincipled):
                break

        bsdf_node = node
        color_input_node = bsdf_node.inputs["Base Color"].links[0].from_node
        color_input_node_socket = (
            bsdf_node.inputs["Base Color"].links[0].from_socket.identifier
        )  # use identifier, names are not unique!
        material.node_tree.links.new(color_input_node.outputs[color_input_node_socket], multiply_node.inputs["Color1"])

        # map the output of the multiply node to the BSDF base color
        material.node_tree.links.new(bsdf_node.inputs["Base Color"], multiply_node.outputs["Color"])

        # disable actual mesh displacements as they change the geometry of the surface
        # and are not used in collision checking, which can cause the cloth to become 'invisible' in renders
        material.cycles.displacement_method = "BUMP"

        # remove existing materials
        object.data.materials.clear()

        object.data.materials.append(material)
    else:
        base_hsv = sample_hsv_color()
        base_rgb = hsv_to_rgb(base_hsv)
        ab.add_material(object, color=base_rgb)

    return object


if __name__ == "__main__":

    PolyhavenMaterials.assets_json_path = "assets.json"
    # select the default cube
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
    new_cube = bpy.context.object

    add_texture_to_object(new_cube)
    print("done")
