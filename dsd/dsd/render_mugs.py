import datetime

import bpy
import numpy as np
from mathutils import Vector

from dsd import DATA_DIR
from dsd.renderer import CyclesRendererConfig, render_scene

# delete default cube
bpy.ops.object.delete(use_global=False)

# add a plane with size 2x2
bpy.ops.mesh.primitive_plane_add(size=2)

# load the mug mesh
mug_path = DATA_DIR / "meshes/mugs/gso_room_essential_white/model.obj"
bpy.ops.import_scene.obj(filepath=str(mug_path), axis_forward="Y", axis_up="Z")
mug_object = bpy.context.selected_objects[0]
mug_object.pass_index = 1  # for segmentation hacky rendering


# output dir
output_dir = DATA_DIR / "renders" / "mugs" / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def sample_point_in_capped_ball(radius, min_height):
    x, y = np.random.uniform(-radius, radius, size=2)
    z = np.random.uniform(min_height, radius)
    return np.array([x, y, z])


def determine_pose_of_camera_looking_at_point(
    camera: bpy.types.Camera, camera_position: np.ndarray, look_at_point: np.ndarray
):
    direction = look_at_point - camera_position
    direction /= np.linalg.norm(direction)
    camera_direction = Vector(direction)
    camera.rotation_euler = camera_direction.to_track_quat("-Z", "Y").to_euler()
    camera.location = Vector(camera_position)
    bpy.context.view_layer.update()  # update the scene to propagate the new camera location & orientation


def create_camera():
    horizontal_fov = 100
    horizontal_resolution = 512
    vertical_resolution = 512

    camera = bpy.data.objects["Camera"]

    # Set the camera intrinsics
    # cf https://docs.blender.org/manual/en/latest/render/cameras.html for more info.

    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.type = "PERSP"
    camera.data.angle = np.pi / 180 * horizontal_fov
    camera.data.lens_unit = "FOV"
    image_width, image_height = horizontal_resolution, vertical_resolution
    scene = bpy.context.scene
    scene.render.resolution_x = image_width
    scene.render.resolution_y = image_height

    return camera


n_renders = 10
for i in range(n_renders):
    # set the camera to a random position
    camera = create_camera()
    camera_position = sample_point_in_capped_ball(0.6, 0.2)
    determine_pose_of_camera_looking_at_point(camera, camera_position, np.array([0, 0, 0]))
    render_scene(render_config=CyclesRendererConfig(), output_dir=str(output_dir / f"{i:03d}"))

    # save the pose of the mug in the camera frame
    mug_pose = np.eye(4)
    mug_pose[:3, 3] = mug_object.location
    mug_pose[:3, :3] = mug_object.rotation_euler.to_matrix()

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = camera.location
    camera_pose[:3, :3] = camera.rotation_euler.to_matrix()

    mug_in_camera_frame = np.linalg.inv(camera_pose) @ mug_pose
    print(mug_in_camera_frame)
    np.save(output_dir / f"{i:03d}" / "mug_pose_in_camera_frame.npy", mug_in_camera_frame)
