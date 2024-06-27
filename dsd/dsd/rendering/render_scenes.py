import dataclasses
import datetime
import json
import random

import bpy
import numpy as np
from airo_blender.materials import add_material
from mathutils import Vector
from tqdm import tqdm

from dsd import DATA_DIR
from dsd.rendering.blender.keypoint_annotator import annotate_keypoints
from dsd.rendering.blender.renderer import CyclesRendererConfig, render_scene


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
    horizontal_fov = (
        45  # vertical FOV realsense is 42 # https://www.framos.com/en/products/depth-camera-d415-camera-only-20801
    )
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


def add_lighting():
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


# need to make configurable:
# - the camera extrinsics (radius of the sphere)
# - mesh directory
# - target directory
# - to render the keypoints or not.
# - number of renders per mesh


@dataclasses.dataclass
class RenderConfig:
    mesh_directory: str
    output_directory: str
    table_scale_range: tuple = (0.2, 0.8)
    render_keypoints: bool = False
    n_renders_per_mesh: int = 25
    camera_minimum_distance: float = 0.3
    camera_maximum_distance: float = 0.7
    camera_minimum_height: float = 0.05
    add_table_to_scene: bool = True
    render_table_without_object: bool = True


def render_scenes(config: RenderConfig):  # noqa C901
    # fix the random seeds to make reproducible renders
    np.random.seed(2024)
    random.seed(2024)
    # delete all objects
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="MESH")
    bpy.ops.object.delete()

    add_lighting()

    if config.add_table_to_scene:
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
    meshes = list(config.mesh_directory.glob("**/*.obj"))
    mesh_object = None
    for mesh in tqdm(meshes):

        if mesh_object is not None:
            bpy.data.objects.remove(mesh_object, do_unlink=True)

        bpy.ops.import_scene.obj(filepath=str(mesh), axis_forward="Y", axis_up="Z")
        mesh_object = bpy.context.selected_objects[0]
        # rename the object
        mesh_object.name = "object"

        # load the 3D keypoints
        keypoints_3D_dict = json.load(open(str(mesh).split(".")[0] + "_keypoints.json", "r"))

        # if tshirt mesh -> solidify the mesh a bit to give it thickness and make it visible on the depth map
        if "tshirt" in str(config.mesh_directory).lower():
            # select the mesh object
            TSHIRT_THICKNESS = 0.02
            bpy.context.view_layer.objects.active = mesh_object

            bpy.ops.object.modifier_add(type="SOLIDIFY")
            bpy.context.object.modifiers["Solidify"].thickness = TSHIRT_THICKNESS
            bpy.context.object.modifiers["Solidify"].offset = 0.0
            bpy.ops.object.modifier_apply(modifier="Solidify")

            # move the object up a bit
            mesh_object.location[2] += TSHIRT_THICKNESS / 2
            bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

            # also move up the keypoints in that case
            for kp_name, kp in keypoints_3D_dict.items():
                keypoints_3D_dict[kp_name][2] += TSHIRT_THICKNESS / 2

        # make the object white
        add_material(
            mesh_object,
            [
                1.0,
            ]
            * 3,
            roughness=1.0,
        )
        mesh_object.pass_index = 1  # for segmentation hacky rendering

        # shade smooth
        bpy.ops.object.shade_smooth()

        mesh_output_dir = config.output_directory / mesh.stem
        mesh_output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(config.n_renders_per_mesh):

            # hack to have same random numbers with and without table
            # should fix the see for each scene in the future to avoid these kind of issues...

            x = np.random.uniform(*config.table_scale_range)
            y = np.random.uniform(*config.table_scale_range)
            if config.add_table_to_scene:
                table.scale = (x, y, 1)

            # apply scale to the object
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

            for j in range(20):
                if j == 19:
                    print("Could not find a valid camera position after 20 iterations, skipping this mug.")
                    break

                # set the camera to a random position
                camera = create_camera()
                camera_position = sample_point_in_capped_ball(
                    config.camera_maximum_distance, config.camera_minimum_distance, config.camera_minimum_height
                )

                # randomize the looking at position around the mug.
                looking_at_position = np.random.uniform(-0.1, 0.1, 3)
                looking_at_position[2] = 0.05
                determine_pose_of_camera_looking_at_point(camera, camera_position, looking_at_position)

                # add slight random rotation to camera
                # camera.rotation_euler += np.random.uniform(-np.pi / 36, np.pi / 36, 3)

                # clear the output dir
                if (mesh_output_dir / f"{i:03d}").exists():
                    import shutil

                    shutil.rmtree(mesh_output_dir / f"{i:03d}")
                render_scene(
                    render_config=CyclesRendererConfig(num_samples=8), output_dir=str(mesh_output_dir / f"{i:03d}")
                )
                # check if the segmentation mask does not border on the image edges
                # if this is not the case, the object is entirely within view
                from PIL import Image

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

            ## save all data

            # save an image of the table without the object

            if config.add_table_to_scene and config.render_table_without_object:
                # temporarily hide the object
                mesh_object.hide_render = True
                # set table pass index to 1
                table.pass_index = 1
                render_scene(
                    render_config=CyclesRendererConfig(num_samples=4, render_rgb=True, render_segmentation=True),
                    output_dir=str(mesh_output_dir / f"{i:03d}" / "table_only"),
                )
                mesh_object.hide_render = False
                table.pass_index = 0

            # save the pose of the mug in the camera frame
            object_pose = np.eye(4)
            object_pose[:3, 3] = mesh_object.location
            object_pose[:3, :3] = mesh_object.rotation_euler.to_matrix()

            camera_pose = np.eye(4)
            camera_pose[:3, 3] = camera.location
            camera_pose[:3, :3] = camera.rotation_euler.to_matrix()

            mug_in_camera_frame = np.linalg.inv(camera_pose) @ object_pose
            np.save(mesh_output_dir / f"{i:03d}" / "object_pose_in_camera_frame.npy", mug_in_camera_frame)

            # get the 2D keypoint

            # if tshirt mesh, undo the solidify modifier before determining the visibility of the 2D keypoints
            if "tshirt" in str(config.mesh_directory).lower():
                # select the mesh object
                bpy.context.view_layer.objects.active = mesh_object
                bpy.ops.object.modifier_remove(modifier="Solidify")

            keypoints_2d = annotate_keypoints(keypoints_3D_dict, camera)

            # plot the keypoints on the image
            if config.render_keypoints:
                from PIL import Image, ImageDraw

                img = Image.open(mesh_output_dir / f"{i:03d}" / "rgb.png")
                draw = ImageDraw.Draw(img)
                for kp_name, (u, v, visibility) in keypoints_2d.items():
                    if visibility < 1.5:
                        draw.ellipse((u - 5, v - 5, u + 5, v + 5), fill=(100, 0, 0))
                    else:
                        draw.ellipse((u - 5, v - 5, u + 5, v + 5), fill=(255, 0, 0))
                img.save(mesh_output_dir / f"{i:03d}" / "annotated_rgb.png")

            # save as json
            with open(mesh_output_dir / f"{i:03d}" / "keypoints.json", "w") as f:
                json.dump(keypoints_2d, f)

            # save the blend file
            bpy.ops.wm.save_as_mainfile(filepath=str(mesh_output_dir / f"{i:03d}" / "scene.blend"))


if __name__ == "__main__":  # noqa
    np.random.seed(2024)
    random.seed(2024)

    ### WITH table configs

    # # mugs
    config = RenderConfig(
        mesh_directory=DATA_DIR / "meshes/mugs/objaverse-mugs-filtered/",
        output_directory=DATA_DIR / "scenes" / "mugs" / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        render_keypoints=False,
        n_renders_per_mesh=25,
    )
    render_scenes(config)

    # # shoes

    config = RenderConfig(
        mesh_directory=DATA_DIR / "meshes/shoes/GSO-labeled",
        output_directory=DATA_DIR / "scenes" / "shoes" / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        render_keypoints=True,
        n_renders_per_mesh=12,
        table_scale_range=(0.3, 1.0),
        camera_minimum_distance=0.4,
        camera_maximum_distance=0.9,
        camera_minimum_height=0.2,
    )

    # tshirts

    # config = RenderConfig(
    #     mesh_directory=DATA_DIR / "meshes/tshirts/processed_meshes/",
    #     output_directory=DATA_DIR / "scenes" / "tshirts" / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    #     render_keypoints=True,
    #     n_renders_per_mesh=10,
    #     table_scale_range=(0.7, 1.3),
    #     camera_minimum_distance=1.0,
    #     camera_maximum_distance=1.8, # matches the max 1m distance of a zed camera with FOV 67 degrees
    #     camera_minimum_height=0.5,
    # )

    ### WO table configs

    # # mugs
    config = RenderConfig(
        mesh_directory=DATA_DIR / "meshes/mugs/objaverse-mugs-filtered/",
        output_directory=DATA_DIR / "scenes" / "mugs" / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        render_keypoints=False,
        n_renders_per_mesh=25,
        add_table_to_scene=False,
    )

    # # shoes
    # render_scenes(config)

    config = RenderConfig(
        mesh_directory=DATA_DIR / "meshes/shoes/GSO-labeled/",
        output_directory=DATA_DIR / "scenes" / "shoes" / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        n_renders_per_mesh=12,
        table_scale_range=(0.3, 1.0),
        camera_minimum_distance=0.4,
        camera_maximum_distance=0.9,
        camera_minimum_height=0.2,
        add_table_to_scene=False,
    )
    # render_scenes(config)

    # tshirts

    config = RenderConfig(
        mesh_directory=DATA_DIR / "meshes/tshirts/processed_meshes/",
        output_directory=DATA_DIR / "scenes" / "tshirts" / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        n_renders_per_mesh=10,
        table_scale_range=(0.7, 1.3),
        camera_minimum_distance=1.0,
        camera_maximum_distance=1.8,  # matches the max 1m distance of a zed camera with FOV 67 degrees
        camera_minimum_height=0.5,
        add_table_to_scene=False,
    )
    render_scenes(config)
