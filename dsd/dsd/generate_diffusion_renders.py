import shutil

import tqdm

from dsd import DATA_DIR
from dsd.diffusion_rendering import (
    ControlNetFromCannyRenderer,
    ControlNetFromDepthRenderer,
    ControlnetfromHEDRenderer,
    ControlNetFromNormalsRenderer,
    DiffusionRenderInputImages,
)

# the source directory

# /....
# / ...
# rgb.png
# depth_image.png
# normal_image.png
# segmentation.png


source_directory = DATA_DIR / "renders" / "mugs" / "small"
target_directory = DATA_DIR / "diffusion_renders" / "mugs" / "small"
# clear if exists
if target_directory.exists():
    shutil.rmtree(target_directory)
target_directory.mkdir(parents=True, exist_ok=True)

num_images_per_prompt = 2
diffusion_renderers = [
    (ControlNetFromDepthRenderer, {"num_images_per_prompt": num_images_per_prompt}),
    (ControlnetfromHEDRenderer, {"num_images_per_prompt": num_images_per_prompt}),
    (ControlNetFromNormalsRenderer, {"num_images_per_prompt": num_images_per_prompt}),
    (ControlNetFromCannyRenderer, {"num_images_per_prompt": num_images_per_prompt}),
]
prompts = ["a colorful mug", "a blue mug on a metallic surface"]

rgb_image_paths = list(source_directory.glob("**/rgb.png"))
image_dirs = [p.parent for p in rgb_image_paths]


for renderer in tqdm.tqdm(diffusion_renderers):
    renderer, kwargs = renderer
    renderer = renderer(**kwargs)
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
        for prompt in prompts:
            renderer_image_target_dir = image_target_dir / renderer.__class__.__name__
            renderer_image_target_dir.mkdir(parents=True, exist_ok=True)
            output_images = renderer(prompt, input_images)
            for i, image in enumerate(output_images):
                image.save(renderer_image_target_dir / f"{prompt}_{i}.png")
