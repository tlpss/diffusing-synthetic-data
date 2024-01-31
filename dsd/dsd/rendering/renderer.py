import dataclasses
import os
import pathlib 

import bpy


@dataclasses.dataclass
class RendererConfig:
    exposure_min: float = 0.0
    exposure_max: float = 0.0
    device: str = "GPU"
    render_rgb: bool = True
    render_depth: bool = True
    render_segmentation: bool = True
    render_normal: bool = False


@dataclasses.dataclass
class CyclesRendererConfig(RendererConfig):
    num_samples: int = 32
    denoise: bool = True


@dataclasses.dataclass
class EeveeRendererConfig(RendererConfig):
    pass


def render_scene(render_config: RendererConfig, output_dir: str):  # noqa: C901
    scene = bpy.context.scene

    if isinstance(render_config, CyclesRendererConfig):
        scene.render.engine = "CYCLES"
        scene.cycles.use_denoising = render_config.denoise
        scene.cycles.samples = render_config.num_samples
        scene.cycles.device = render_config.device

    elif isinstance(render_config, EeveeRendererConfig):
        scene.render.engine = "BLENDER_EEVEE"
    else:
        raise NotImplementedError(f"Renderer config {render_config} not implemented")

    # scene.view_settings.exposure = np.random.uniform(render_config.exposure_min, render_config.exposure_max)
    scene.view_settings.gamma = 1.0

    # Make a directory to organize all the outputs
    os.makedirs(output_dir, exist_ok=True)

    scene.use_nodes = True

    # remove any existing NodeOutputFile nodes from the scene
    # to enable calling this render method multiple times in the same blender session
    if scene.node_tree.nodes and len(scene.node_tree.nodes) > 0:
        for node in scene.node_tree.nodes:
            if node.name == "Render Layers":
                continue
            scene.node_tree.nodes.remove(node)

    # Add a file output node to the scene
    tree = scene.node_tree
    links = tree.links
    nodes = tree.nodes
    node = nodes.new("CompositorNodeOutputFile")
    node.location = (500, 200)
    node.base_path = output_dir

    if render_config.render_rgb:
        slot_image = node.file_slots["Image"]
        slot_image.path = "rgb"
        slot_image.format.color_mode = "RGB"
        slot_image.use_node_format = False
        slot_image.save_as_render = True
        render_layers_node = nodes["Render Layers"]
        links.new(render_layers_node.outputs["Image"], node.inputs["Image"])


    if render_config.render_segmentation:
        scene.view_layers["ViewLayer"].use_pass_object_index = True
        segmentation_name = "segmentation"
        node.file_slots.new(segmentation_name)
        slot_segmentation = node.file_slots[segmentation_name]

        # slot_segmentation.path = f"{random_seed:08d}_segmentation"
        slot_segmentation.format.color_mode = "BW"
        slot_segmentation.use_node_format = False
        slot_segmentation.save_as_render = False


        # Other method, use the mask ID node
        mask_id_node = nodes.new("CompositorNodeIDMask")
        mask_id_node.index = 1  # TODO: make this configurable, instead of hardcoding cloth ID
        mask_id_node.location = (300, 200)
        links.new(render_layers_node.outputs["IndexOB"], mask_id_node.inputs[0])
        links.new(mask_id_node.outputs[0], node.inputs[slot_segmentation.path])

    if render_config.render_depth:
        normalization_node = nodes.new("CompositorNodeNormalize")

        camera = bpy.context.scene.camera
        clip_node = nodes.new("CompositorNodeMapValue")
        clip_node.use_max = True
        clip_node.max = (camera.data.clip_end,)
        depth_image_name = "depth_image"
        depth_map_name = "depth_map"
        scene.view_layers["ViewLayer"].use_pass_z = True

        node.file_slots.new(depth_image_name)
        slot_depth_image = node.file_slots[depth_image_name]
        slot_depth_image.format.color_mode = "BW"
        slot_depth_image.use_node_format = False
        slot_depth_image.save_as_render = False

        node.file_slots.new(depth_map_name)
        slot_depth_map = node.file_slots[depth_map_name]
        slot_depth_map.format.file_format = "OPEN_EXR"
        slot_depth_map.format.color_depth = "16"
        slot_depth_map.use_node_format = False
        slot_depth_map.save_as_render = False

        render_layers_node = nodes["Render Layers"]
        links.new(render_layers_node.outputs["Depth"], normalization_node.inputs[0])
        links.new(normalization_node.outputs[0], node.inputs[slot_depth_image.path])
        links.new(render_layers_node.outputs["Depth"], clip_node.inputs["Value"])
        links.new(clip_node.outputs["Value"], node.inputs[slot_depth_map.path])

    if render_config.render_normal:
        normal_image_name = "normal_image"
        scene.view_layers["ViewLayer"].use_pass_normal = True

        node.file_slots.new(normal_image_name)
        slot_normal_image = node.file_slots[normal_image_name]
        slot_normal_image.format.color_mode = "RGB"
        slot_normal_image.use_node_format = False
        slot_normal_image.save_as_render = False

        render_layers_node = nodes["Render Layers"]
        links.new(render_layers_node.outputs["Normal"], node.inputs[slot_normal_image.path])

    bpy.ops.render.render(animation=False)

    # # Prevent the 0001 suffix from being added to the file name
    # # Annoying fix, because Blender adds a 0001 suffix to the file name which can't be disabled
    for file in os.listdir(output_dir):
        if pathlib.Path(file).suffix in [".exr",".png",".jpg",".tiff",".tif"]:
            # remove the 001 suffix before the extension
            filename = os.path.join(output_dir, file)
            filename_new = filename.split(".")[0].removesuffix("0001") + "." + filename.split(".")[1]
            os.rename(filename, filename_new)