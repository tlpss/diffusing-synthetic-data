import glob
import json
import os

import bpy


def arrange_meshes_and_keypoints(folder_path):
    grid_spacing = 0.2
    x_offset = 0

    obj_files = glob.glob(os.path.join(folder_path, "**/*.obj"), recursive=True)

    for obj_file in obj_files:
        bpy.ops.import_scene.obj(filepath=str(obj_file), axis_forward="Y", axis_up="Z")
        obj = bpy.context.selected_objects[0]
        obj.location = (x_offset, 0, 0)

        # keypoint_file = obj_file.replace(".obj", "_keypoints.json")
        # keypoints = load_keypoints(keypoint_file)
        # apply_keypoints_to_mesh(obj, keypoints)

        x_offset += grid_spacing


def load_keypoints(keypoint_file):
    with open(keypoint_file, "r") as f:
        keypoints = json.load(f)
    return keypoints


def apply_keypoints_to_mesh(mesh_object, keypoints):
    for keypoint_name, coordinates in keypoints.items():
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.01, location=coordinates)
        keypoint_sphere = bpy.context.object
        keypoint_sphere.name = f"keypoint_{keypoint_name}"
        keypoint_sphere.parent = mesh_object


if __name__ == "__main__":
    from dsd import DATA_DIR

    folder_path = DATA_DIR / "meshes" / "shoes" / "GSO-shoes"
    arrange_meshes_and_keypoints(folder_path)
