import random
import shutil

import tqdm

from dsd.diffusion_rendering import DiffusionRenderInputImages

# the source directory

# /....
# / ...
# rgb.png
# depth_image.png
# normal_image.png
# segmentation.png


def generate_diffusion_renders(
    source_directory, target_directory, diffusion_renderers, prompts, num_prompts_per_scene=None
):
    rgb_image_paths = list(source_directory.glob("**/rgb.png"))
    image_dirs = [p.parent for p in rgb_image_paths]

    for renderer in tqdm.tqdm(diffusion_renderers):
        renderer, kwargs = renderer
        renderer = renderer(**kwargs)
        # disable NSFW filter
        renderer.pipe.safety_checker = None

        renderer.pipe.set_progress_bar_config(disable=True)
        for image_dir in tqdm.tqdm(image_dirs, desc="diffusion renderers"):
            relative_path_to_source_dir = image_dir.relative_to(source_directory)
            image_target_dir = target_directory / relative_path_to_source_dir
            image_target_dir.mkdir(parents=True, exist_ok=True)
            # copy the orignal images
            blender_image_target_dir = image_target_dir / "original"
            blender_image_target_dir.mkdir(parents=True, exist_ok=True)
            for image_path in image_dir.glob("*"):
                shutil.copy(image_path, blender_image_target_dir)
            input_images = DiffusionRenderInputImages.from_render_dir(image_dir)

            prompts_to_render = random.sample(prompts, num_prompts_per_scene) if num_prompts_per_scene else prompts
            for prompt in prompts_to_render:
                renderer_image_target_dir = image_target_dir / renderer.get_logging_name()
                renderer_image_target_dir.mkdir(parents=True, exist_ok=True)
                output_images = renderer(prompt, input_images)
                for i, image in enumerate(output_images):
                    image.save(renderer_image_target_dir / f"{prompt}_{i}.png")
