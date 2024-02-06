import random

from dsd import DATA_DIR
from dsd.diffusion_rendering import (
    ControlNetFromCannyRenderer,
    ControlNetFromDepthRenderer,
    ControlnetfromHEDRenderer,
    ControlNetFromNormalsRenderer,
    SD2FromDepthRenderer,
    SD2InpaintingRenderer,
    SDXLControlNetFromCannyRenderer,
    SDXLControlNetFromDepthRenderer,
)
from dsd.generate_diffusion_renders import generate_diffusion_renders

source_directory = DATA_DIR / "renders" / "mugs" / "10"
target_directory = DATA_DIR / "diffusion_renders" / "mugs" / "run_3"

num_images_per_prompt = 1
diffusion_renderers = [
    (ControlnetfromHEDRenderer, {"num_images_per_prompt": num_images_per_prompt}),
    (ControlNetFromNormalsRenderer, {"num_images_per_prompt": num_images_per_prompt}),
    (ControlNetFromCannyRenderer, {"num_images_per_prompt": num_images_per_prompt}),
    (ControlNetFromDepthRenderer, {"num_images_per_prompt": num_images_per_prompt}),
    (SDXLControlNetFromCannyRenderer, {"num_images_per_prompt": num_images_per_prompt}),
    (SDXLControlNetFromDepthRenderer, {"num_images_per_prompt": num_images_per_prompt}),
    (SD2FromDepthRenderer, {"num_images_per_prompt": num_images_per_prompt}),
    (SD2InpaintingRenderer, {"num_images_per_prompt": num_images_per_prompt}),
]
prompts = [
    "a striped mug on a wooden table",
    "a blue mug on a metallic surface",
    "a mug on a sofa",
    "a rainbow mug on a white desk",
    "a blue mug on a ceramic table",
    "a black mug on a kitchen table",
    "a mug with a picture of a dog on a table",
    "an orange mug on an outdoor table",
    "a coffee mug on a kitchen counter",
    "a mug",
]


LIGHTINGS = ["", "ambient light", "studio lighting", "natural light"]
STYLE_MEDIUM = ["", "Photorealistic", "Photography, 4K"]

random.seed(2024)
prompts = [prompt + ", " + random.choice(LIGHTINGS) + ", " + random.choice(STYLE_MEDIUM) for prompt in prompts]
# remove double commas
prompts = [prompt.replace(", ,", ",") for prompt in prompts]
# remove trailing comma
prompts = [prompt.replace(", ", "") for prompt in prompts]

print(prompts)
prompts = prompts[:2]
generate_diffusion_renders(source_directory, target_directory, diffusion_renderers, prompts)
