import json
import pathlib

import bpy
from airo_blender import add_material


def srgb2lin(s):
    if s <= 0.0404482362771082:
        lin = s / 12.92
    else:
        lin = pow(((s + 0.055) / 1.055), 2.4)
    return lin


colors = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (70, 240, 240),
    (240, 50, 230),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (0, 0, 128),
    (128, 128, 128),
    (255, 255, 255),
    (0, 0, 0),
]
colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]

# blender user the linear color space by default
# but our colors were defined in sRGB space
# so we need to convert them to linear space before using them in blender!
# cf https://blender.stackexchange.com/questions/153048/blender-2-8-python-input-rgb-doesnt-match-hex-color-nor-actual-color
linear_colors = [(srgb2lin(r), srgb2lin(g), srgb2lin(b)) for r, g, b in colors]


def visualize_annotated_mesh(mesh_path, keypoints_path):
    # delete the default cube
    bpy.ops.object.select_all(action="SELECT")
    bpy.data.objects["Cube"].select_set(True)
    bpy.ops.object.delete()

    # load the mesh into the scene
    bpy.ops.import_scene.obj(filepath=mesh_path, axis_forward="Y", axis_up="Z")

    # read the keypoints json file
    # for each keypoint, create a sphere at the location of the keypoint with a radius of 0.01

    kp_dict = json.load(open(keypoints_path))
    for i, (kp_name, kp_loc) in enumerate(kp_dict.items()):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.02, location=kp_loc)
        # set the color of the sphere

        add_material(bpy.context.object, color=linear_colors[i])
        bpy.context.object.name = kp_name
    # save the scene as a .blend file

    # move camera to new location
    bpy.ops.object.camera_add(location=(1, -1, 0.3), rotation=(1.6, 0, 0.8))
    current_dir = pathlib.Path(__file__).parent
    bpy.ops.wm.save_as_mainfile(filepath=str(current_dir / "annotated_mesh.blend"))


if __name__ == "__main__":
    # mesh_path = "/home/tlips/Documents/diffusing-synthetic-data/data/meshes/shoes/GSO-labeled/11pro_SL_TRX_FG.obj"
    # keypoints_path = "/home/tlips/Documents/diffusing-synthetic-data/data/meshes/shoes/GSO-labeled/11pro_SL_TRX_FG_keypoints.json"

    # mesh_path = "/home/tlips/Documents/diffusing-synthetic-data/data/meshes/tshirts/processed_meshes/000026.obj"
    # keypoints_path = "/home/tlips/Documents/diffusing-synthetic-data/data/meshes/tshirts/processed_meshes/000026_keypoints.json"

    # mug
    mesh_path = "/home/tlips/Documents/diffusing-synthetic-data/data/meshes/mugs/objaverse-mugs-filtered/1dd9b373e3084cc0914a580ce5728cf8/1dd9b373e3084cc0914a580ce5728cf8.obj"
    keypoints_path = "/home/tlips/Documents/diffusing-synthetic-data/data/meshes/mugs/objaverse-mugs-filtered/1dd9b373e3084cc0914a580ce5728cf8/1dd9b373e3084cc0914a580ce5728cf8_keypoints.json"
    visualize_annotated_mesh(mesh_path, keypoints_path)
