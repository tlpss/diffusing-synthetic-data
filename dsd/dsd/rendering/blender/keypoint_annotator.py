import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from dsd.rendering.blender.visibility_check import is_point_in_camera_frustum, is_point_occluded_for_scene_camera


def annotate_keypoints(keypoints_3D_dict: dict, camera: bpy.types.Object) -> dict:
    """converts 3D keypoints (in world frame) to 2D keypoints (in camera frame)"""

    kp_2D_dict = {}
    for kp_name, kp_3d in keypoints_3D_dict.items():
        # determine if it is visible from the camera
        visibility = 2.0
        kp_3d = Vector(kp_3d)
        if is_point_occluded_for_scene_camera(kp_3d, helper_cube_scale=0.01):
            visibility = 1.0

        elif not is_point_in_camera_frustum(kp_3d, camera):
            visibility = 0.0

        kp_2D = np.array(world_to_camera_view(bpy.context.scene, camera, kp_3d))

        # flip y-axis to match the image coordinate system
        kp_2D[1] = 1 - kp_2D[1]

        # scale to pixel coords
        kp_2D[0] *= bpy.context.scene.render.resolution_x
        kp_2D[1] *= bpy.context.scene.render.resolution_y

        u, v = kp_2D[0], kp_2D[1]

        kp_2D_dict[kp_name] = (u, v, visibility)

    return kp_2D_dict
