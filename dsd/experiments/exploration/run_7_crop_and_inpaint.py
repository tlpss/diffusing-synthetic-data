import random

from dsd import DATA_DIR
from dsd.cropped_diffusion_rendering import CropAndInpaintRenderer, CroppedRenderer
from dsd.diffusion_rendering import SD2InpaintingRenderer, SD15RealisticCheckpointControlNetFromDepthRenderer
from dsd.generate_diffusion_renders import generate_crop_inpaint_diffusion_renders

# gemini Ultra prompting.
colors = [
    "deep blue",
    "soft pink",
    "bright yellow",
    "forest green",
    "vibrant orange",
    "dark red",
    "light blue",
    "dark purple",
    "bright white",
    "black",
    "gray",
    "brown",
    "beige",
    "light green",
    "light yellow",
    "light purple",
    "light orange",
    "light red",
    "light pink",
    "light brown",
]

materials = [
    "ceramic",
    "ceramic",
    "ceramic",
    "porcelain",
    "porcelain",
    "porcelain",
    "porcelain",
    "glass",
    "stainless steel",
    "",
    "",
    "",
    "",
]

textures = [
    "smooth",
    "glazed",
    "striped",
    "checkered",
    "speckled",
    "hand-painted",
    "matte",
    "glossy"
    # ... Add more textures
]

table_descriptions = [
    "rustic farmhouse table with natural wood grain",
    "modern glass-topped table with sleek metal legs",
    "antique wooden table with ornate carvings",
    "polished marble table with a luxurious feel",
    "colorful outdoor patio table with a mosaic design",
    "small, round cafe table with a wrought-iron base",
    "child's play table with a brightly painted surface",
    "cluttered office desk with stacked papers and books",
    "sunny windowsill serving as an impromptu table",
    "weathered picnic table in a park setting",
    "minimalist coffee table with clean lines",
    "sturdy workbench with visible wear and tear",
    "vintage kitchen table with a checkered tablecloth",
    "folding card table with a felt surface",
    "side table crafted from a repurposed tree stump",
    "granite countertop doubling as a table surface",
    "elegant dining table set with fine china",
    "beachside table made of driftwood",
    "low Japanese-style table for floor seating",
    "travel table with collapsible legs",
    "round wicker table with a woven texture",
    "industrial-style table with exposed metal pipes",
    "concrete table with a modern, minimalist look",
    "bamboo table with a natural, sustainable feel",
    "mid-century modern table with tapered legs",
    "antique writing desk with hidden compartments",
    "vintage suitcase repurposed as a quirky table",
    "kitchen island with a butcher block top",
    "wrought-iron bistro table on a balcony",
    "plastic folding table for outdoor events",
    "tree trunk slice transformed into a side table",
    "tiled table with a vibrant Mediterranean design",
    "hand-painted table with a whimsical folk art style",
    "mirrored tabletop with a glamorous effect",
    "table with a base made of stacked books",
    "acrylic table with a clear, contemporary feel",
    "table built from reclaimed pallets",
    "retro diner booth with a built-in table",
    "ottoman with a removable tray top for table use",
    "chessboard table with inlaid wood squares",
    "lab table with a durable, chemical-resistant surface",
    "terrarium table with a glass top and plants inside",
    "extendable dining table for large gatherings",
    "vintage sewing machine base converted into a table",
    "table sculpted from a single piece of wood",
    "table adorned with a colorful stained glass top",
    "antique vanity transformed into a side table",
    "table with legs made of stacked tires",
    "blanket draped over crates serving as a makeshift table",
    "large rock with a flat surface used as a natural table",
]

LIGHTINGS = ["", "ambient light", "studio lighting", "natural light"]
STYLE_MEDIUM = ["", "RAW", "Photorealistic", "Photography, 4K"]

random.seed(2024)

prompts = []
for _ in range(1000):
    color = random.choice(colors)
    material = random.choice(materials)
    texture = random.choice(textures)
    table_desc = random.choice(table_descriptions)
    prompt = f"A {color} {material} mug with a {texture} finish"
    prompts.append(prompt)


prompts = [prompt + ", " + random.choice(LIGHTINGS) + ", " + random.choice(STYLE_MEDIUM) for prompt in prompts]
# remove double commas
prompts = [prompt.replace(", ,", ",") for prompt in prompts]
# remove trailing comma at the end of the string
prompts = [prompt.rstrip(", ") for prompt in prompts]


background_prompts = []
for x in table_descriptions:
    background_prompts.append(f"a {x}, " + random.choice(LIGHTINGS) + ", " + random.choice(STYLE_MEDIUM))
background_prompts = [prompt.replace(", ,", ",") for prompt in background_prompts]
background_prompts = [prompt.rstrip(", ") for prompt in background_prompts]

# # clear if exists
# if target_directory.exists():
#     shutil.rmtree(target_directory)
# target_directory.mkdir(parents=True, exist_ok=True)


source_directory = DATA_DIR / "renders" / "mugs" / "objaverse-1000"
target_directory = DATA_DIR / "diffusion_renders" / "mugs" / "run_7"

num_images_per_prompt = 2
num_prompts_per_scene = 1
diffusion_renderer = SD15RealisticCheckpointControlNetFromDepthRenderer(num_images_per_prompt)
cropped_renderer = CroppedRenderer(diffusion_renderer)
inpaint_renderer = SD2InpaintingRenderer(num_images_per_prompt=num_images_per_prompt)
crop_and_inpaint_renderer = CropAndInpaintRenderer(cropped_renderer, inpaint_renderer)
generate_crop_inpaint_diffusion_renders(
    source_directory,
    target_directory,
    [crop_and_inpaint_renderer],
    prompts,
    background_prompts,
    num_prompts_per_scene=num_prompts_per_scene,
)
