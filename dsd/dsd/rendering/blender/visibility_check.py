""" code to check if blender objects, points are occluded in a camera view

copied from https://github.com/tlpss/synthetic-cloth-data/blob/main/synthetic-cloth-data/synthetic_cloth_data/synthetic_images/scene_builder/utils/visible_vertices.py

"""
import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

DEBUG = False


def is_point_occluded_for_scene_camera(co: Vector, helper_cube_scale: float = 0.0001) -> bool:
    """Checks if a point is occluded by objects in the scene w.r.t. the camera.

    Args:
        co (Vector): the world space x, y and z coordinates of the point.

    Returns:
        boolean: visibility
    """
    co = Vector(co)

    bpy.context.view_layer.update()  # ensures camera matrix is up to date
    scene = bpy.context.scene
    camera_obj = scene.camera  # bpy.types.Object

    # add small cube around coord to make sure the ray will intersect
    # as the ray_cast is not always accurate
    # cf https://blender.stackexchange.com/a/87755
    bpy.ops.mesh.primitive_cube_add(location=co, scale=(helper_cube_scale, helper_cube_scale, helper_cube_scale))
    cube = bpy.context.object
    direction = co - camera_obj.location
    hit, location, _, _, _, _ = scene.ray_cast(
        bpy.context.view_layer.depsgraph,
        origin=camera_obj.location + direction * 0.0001,  # avoid self intersection
        direction=direction,
    )

    if DEBUG:
        print(f"hit location: {location}")
        bpy.ops.mesh.primitive_ico_sphere_add(
            location=co, scale=(helper_cube_scale, helper_cube_scale, helper_cube_scale)
        )

    # remove the auxiliary cube
    if not DEBUG:
        bpy.data.objects.remove(cube, do_unlink=True)

    if not hit:
        raise ValueError("No hit found, this should not happen as the ray should always hit the vertex itself.")
    # if the hit is the vertex itself, it is not occluded
    if (location - co).length < helper_cube_scale * 2:
        return False
    return True


def is_object_occluded_for_scene_camera(obj: bpy.types.Object) -> bool:
    """Checks if all vertices of an object are occluded by objects in the scene w.r.t. the camera.

    Args:
        obj (bpy.types.Object): the object.

    Returns:
        boolean: visibility
    """
    for vertex in obj.data.vertices:
        coords = obj.matrix_world @ vertex.co
        if not is_point_occluded_for_scene_camera(coords):
            return False
    return True


def is_point_in_camera_frustum(point: Vector, camera: bpy.types.Object) -> bool:
    """Check if a point is in the camera frustum."""
    # Project the point
    scene = bpy.context.scene
    projected_point = world_to_camera_view(scene, camera, point)
    # Check if the point is in the frustum
    return (
        0 <= projected_point[0] <= 1
        and 0 <= projected_point[1] <= 1
        and camera.data.clip_start <= projected_point[2] <= camera.data.clip_end
    )
