import datetime
import json
import random

import bpy
import numpy as np
from airo_blender.materials import add_material
from mathutils import Vector
from PIL import Image

from dsd import DATA_DIR
from dsd.rendering.keypoint_annotator import annotate_keypoints
from dsd.rendering.renderer import CyclesRendererConfig, render_scene


def sample_point_in_capped_ball(max_radius, min_radius, min_height):
    max_iterations = 20
    for _ in range(max_iterations):
        phi, theta = np.random.uniform(0, 2 * np.pi), np.random.uniform(0, np.pi / 2)
        r = np.random.uniform(min_radius, max_radius)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        if z > min_height:
            return np.array([x, y, z])

    raise ValueError(f"Could not find a point in the capped ball after {max_iterations} iterations.")


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
    horizontal_fov = 70  # cropped zed camera. vertical fov is 70
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

    # set depth clipping to 3 meters to avoid huge range in depth map output.
    camera.data.clip_end = 3.0

    return camera


if __name__ == "__main__":  # noqa
    XY_SCALE_RANGE = (0.8, 1.2)
    Z_SCALE_RANGE = (0.8, 1.2)

    # fix the random seeds to make reproducible renders
    np.random.seed(2024)
    random.seed(2024)

    n_renders = 20

    # input dir
    mugs_path = DATA_DIR / "meshes/objaverse-mugs/"

    # output dir
    output_dir = DATA_DIR / "renders" / "mugs" / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # delete default cube
    bpy.ops.object.delete(use_global=False)

    # delete default light
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete(use_global=False)

    # add a diffuse lights
    bpy.ops.object.light_add(type="AREA", location=(0, 2, 2), rotation=(-np.pi / 6, 0, 0))
    light = bpy.context.selected_objects[0]
    light.data.energy = 100
    light.data.size = 2

    bpy.ops.object.light_add(type="AREA", location=(0, -2, 2), rotation=(np.pi / 6, 0, 0))
    light = bpy.context.selected_objects[0]
    light.data.energy = 100
    light.data.size = 2

    # add a plane with size 2x2
    # using a cube with scale 2x2x0.01
    bpy.ops.mesh.primitive_cube_add(scale=(1, 1, 0.01), location=(0, 0, -0.011))

    # make the color of the table dark grey
    table = bpy.context.selected_objects[0]
    add_material(
        table,
        [
            0.2,
        ]
        * 3,
        roughness=1.0,
    )

    meshes = list(mugs_path.glob("**/*.obj"))
    mug_object = None
    for mesh in meshes:

        if mug_object is not None:
            bpy.data.objects.remove(mug_object, do_unlink=True)

        bpy.ops.import_scene.obj(filepath=str(mesh), axis_forward="Y", axis_up="Z")
        mug_object = bpy.context.selected_objects[0]
        # make the mug white
        add_material(
            mug_object,
            [
                1.0,
            ]
            * 3,
            roughness=1.0,
        )
        mug_object.pass_index = 1  # for segmentation hacky rendering

        mesh_output_dir = output_dir / mesh.stem
        mesh_output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_renders):
            # scale the mug to a random size
            xy_scale = np.random.uniform(*XY_SCALE_RANGE)
            z_scale = np.random.uniform(*Z_SCALE_RANGE)
            scale = (xy_scale, xy_scale, z_scale)

            mug_object.scale = scale

            table.scale[0] = np.random.uniform(0.1, 0.8)
            table.scale[1] = np.random.uniform(0.1, 0.8)

            # apply scale to the object
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

            for j in range(20):
                if j == 19:
                    print("Could not find a valid camera position after 20 iterations, skipping this mug.")
                    break

                # set the camera to a random position
                camera = create_camera()
                camera_position = sample_point_in_capped_ball(0.8, 0.3, 0.05)

                # randomize the looking at position around the mug.
                looking_at_position = np.random.uniform(-0.1, 0.1, 3)
                looking_at_position[2] = 0.05
                determine_pose_of_camera_looking_at_point(camera, camera_position, looking_at_position)

                # add slight random rotation to camera
                camera.rotation_euler += np.random.uniform(-np.pi / 12, np.pi / 12, 3)

                # clear the output dir
                if (mesh_output_dir / f"{i:03d}").exists():
                    import shutil

                    shutil.rmtree(mesh_output_dir / f"{i:03d}")
                render_scene(
                    render_config=CyclesRendererConfig(num_samples=8), output_dir=str(mesh_output_dir / f"{i:03d}")
                )
                # check if the segmentation mask does not border on the image edges
                # if this is not the case, the object is entirely within view

                segmentation = Image.open(str(mesh_output_dir / f"{i:03d}" / "segmentation.png"))
                # segmentation.save(str(mesh_output_dir / f"{i:03d}" / f"{j}_segmentation.png"))
                segmentation = (np.array(segmentation) > 0) * 1.0
                if (
                    np.any(segmentation[0, :] == 1)
                    or np.any(segmentation[-1, :] == 1)
                    or np.any(segmentation[:, 0] == 1)
                    or np.any(segmentation[:, -1] == 1)
                    or np.sum(segmentation) < 10
                ):
                    print("Object is not (entirely) within view, re-rendering")
                    continue

                else:
                    break
            if j == 19:
                print("Could not find a valid camera position after 20 iterations, skipping this mug.")
                # remove the output dir
                import shutil

                shutil.rmtree(mesh_output_dir / f"{i:03d}")
                continue

            # save the pose of the mug in the camera frame
            mug_pose = np.eye(4)
            mug_pose[:3, 3] = mug_object.location
            mug_pose[:3, :3] = mug_object.rotation_euler.to_matrix()

            camera_pose = np.eye(4)
            camera_pose[:3, 3] = camera.location
            camera_pose[:3, :3] = camera.rotation_euler.to_matrix()

            mug_in_camera_frame = np.linalg.inv(camera_pose) @ mug_pose
            np.save(mesh_output_dir / f"{i:03d}" / "mug_pose_in_camera_frame.npy", mug_in_camera_frame)

            # get the 2D keypoint
            keypoints_3D_dict = json.load(open(str(mesh).split(".")[0] + "_keypoints.json", "r"))
            # scale the keypoints
            for key, value in keypoints_3D_dict.items():
                keypoints_3D_dict[key] = np.array(value) * scale

            keypoints_2d = annotate_keypoints(keypoints_3D_dict, camera)

            # # plot them on the image
            # from PIL import Image, ImageDraw
            # img = Image.open(mesh_output_dir / f"{i:03d}" / "rgb.png")
            # draw = ImageDraw.Draw(img)
            # for kp_name, (u, v, visibility) in keypoints_2d.items():
            #     if visibility < 1.5:
            #         draw.ellipse((u - 5, v - 5, u + 5, v + 5), fill=(100, 0, 0))
            #     else:
            #         draw.ellipse((u - 5, v - 5, u + 5, v + 5), fill=(255, 0, 0))
            # img.save(mesh_output_dir / f"{i:03d}" / "annotated_rgb.png")

            # save as json
            with open(mesh_output_dir / f"{i:03d}" / "keypoints.json", "w") as f:
                json.dump(keypoints_2d, f)
