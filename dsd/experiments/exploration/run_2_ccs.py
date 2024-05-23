from dsd import DATA_DIR
from dsd.diffusion_rendering import ControlNetFromDepthRenderer
from dsd.generate_diffusion_renders import generate_diffusion_renders

source_directory = DATA_DIR / "renders" / "mugs" / "10"
target_directory = DATA_DIR / "diffusion_renders" / "mugs" / "run_2_ccs"

num_images_per_prompt = 4
diffusion_renderers = [
    (
        ControlNetFromDepthRenderer,
        {"num_images_per_prompt": num_images_per_prompt, "controlnet_conditioning_scale": 0.5},
    ),
    (
        ControlNetFromDepthRenderer,
        {"num_images_per_prompt": num_images_per_prompt, "controlnet_conditioning_scale": 1.0},
    ),
    (
        ControlNetFromDepthRenderer,
        {"num_images_per_prompt": num_images_per_prompt, "controlnet_conditioning_scale": 1.3},
    ),
    (
        ControlNetFromDepthRenderer,
        {"num_images_per_prompt": num_images_per_prompt, "controlnet_conditioning_scale": 1.5},
    ),
    (
        ControlNetFromDepthRenderer,
        {"num_images_per_prompt": num_images_per_prompt, "controlnet_conditioning_scale": 2.0},
    ),
    (
        ControlNetFromDepthRenderer,
        {"num_images_per_prompt": num_images_per_prompt, "controlnet_conditioning_scale": 2.5},
    ),
    # (ControlnetfromHEDRenderer, {"num_images_per_prompt": num_images_per_prompt}),
    # (ControlNetFromNormalsRenderer, {"num_images_per_prompt": num_images_per_prompt}),
    # (ControlNetFromCannyRenderer, {"num_images_per_prompt": num_images_per_prompt}),
]
prompts = [
    "a colorful mug",
    # "a blue mug on a metallic surface",
    " a mug on a sofa",
    # "a mug on a surface",
    "a blue mug on a wooden table",
    # "a mug on a table",
    # "a mug with a picture of a dog",
    "an orange mug",
    # " a coffee mug",
    # "a cup",
    # "a cup on a table",
]


generate_diffusion_renders(source_directory, target_directory, diffusion_renderers, prompts)
