import shutil
from dsd import DATA_DIR
from dsd.diffusion_rendering import ControlNetFromDepthRenderer, DiffusionRenderInputImages, ControlnetfromHEDRenderer

# the source directory 

# /....
    # / ...
        # rgb.png
        # depth_image.png
        # normal_image.png
        # segmentation.png


source_directory = DATA_DIR / "renders" / "mugs" / "2024-01-29_12-37-22"
target_directory = DATA_DIR / "diffusion_renders" / "mugs" / "2024-01-29_12-37-22"
# clear if exists
if target_directory.exists():
    shutil.rmtree(target_directory)
target_directory.mkdir(parents=True, exist_ok=True)
diffusion_renderers = [ControlNetFromDepthRenderer(num_images_per_prompt=1), ControlnetfromHEDRenderer(num_images_per_prompt=1)]
prompts = ["a colorful mug"]

rgb_image_paths = list(source_directory.glob("**/rgb.png"))
image_dirs = [p.parent for p in rgb_image_paths]


for renderer in diffusion_renderers:
    for image_dir in image_dirs:
        relative_path_to_source_dir = image_dir.relative_to(source_directory)
        image_target_dir = target_directory / relative_path_to_source_dir
        image_target_dir.mkdir(parents=True, exist_ok=True)
        input_images = DiffusionRenderInputImages.from_render_dir(image_dir)
        for prompt in prompts:
            renderer_image_target_dir = image_target_dir / renderer.__class__.__name__
            renderer_image_target_dir.mkdir(parents=True, exist_ok=True)
            output_images = renderer(prompt, input_images)
            for i, image in enumerate(output_images):
                image.save(renderer_image_target_dir / f"{prompt}_{i}.png")
