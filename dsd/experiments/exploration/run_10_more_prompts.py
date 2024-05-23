import random

from dsd import DATA_DIR
from dsd.diffusion_rendering import SD15RealisticCheckpointControlNetFromDepthRenderer
from dsd.generate_diffusion_renders import generate_diffusion_renders

# gemini Ultra prompting.
mug_descriptions = [
    # Texture, Material, Color, Painting
    "Smooth, ceramic mug with a watercolor floral design in soft pinks and greens",
    "Hand-thrown stoneware mug with a textured geometric pattern and a matte charcoal finish",
    "Glossy porcelain mug with a vintage botanical illustration featuring delicate blues and yellows",
    "Earthenware mug with a crackled glaze in muted earth tones like terracotta and beige",
    "Wooden mug with a natural walnut grain pattern and a simple carved inscription",
    "Brightly colored enamel mug in sunshine yellow with a playful cartoon animal design",
    "Matte white mug with a minimalist line art  drawing in black",
    "Antiqued copper mug with an embossed Celtic knotwork design",
    "Personalized ceramic mug with a full-color landscape photo and a satin finish",
    "Colorful plastic travel mug with a rubberized teal grip and a bold geometric logo",
    "Smooth, speckled stoneware mug with a hand-painted abstract design in shades of teal and rust",
    "Glossy ceramic mug with a detailed pop art portrait in vibrant primary colors",
    "Hand-thrown earthenware mug with a textured swirl pattern in deep blues and greens",
    "Matte black mug with a single white inspirational quote in elegant calligraphy",
    "Vintage porcelain mug with a faded floral pattern in soft pastel colors",
    "Clear glass mug with an etched mountain scene",
    "Colorful ceramic mug with a hand-painted mosaic design in turquoise, orange, and gold",
    "Antiqued pewter mug with an embossed astrological symbol",
    "Personalized ceramic mug with a child's drawing in bold crayon colors",
    "Textured stoneware mug with a drip glaze effect in shades of blue and white",
    "Smooth ceramic mug with a colorful pixel art character",
    "Wooden mug with a natural cherry wood grain and a playful carved animal",
    "Earthenware mug with a speckled glaze in shades of brown and cream",
    "Retro enamel mug in bright red with a humorous meme printed in black",
    "Personalized ceramic mug with a full-color pet portrait and a glossy finish",
    "Matte ceramic mug with a minimalist geometric design in black and white",
    "Hand-painted porcelain mug with a delicate chinoiserie-inspired floral pattern",
    "Earthenware mug with a rough, unglazed surface and a hand-painted abstract design",
    "Smooth ceramic mug with a marbled glaze effect in shades of purple and pink",
    "Vintage metal mug with a humorous retro advertising design",
    "Shiny stainless steel mug with an engraved motivational message",
    "Textured ceramic mug in a deep cobalt blue with a metallic gold accent design",
    "Wooden mug with a natural oak grain pattern and a simple geometric inlay",
    "Ceramic mug with a speckled glaze in shades of gray and a single hand-painted leaf",
    "Colorful plastic mug with a photorealistic still life of fruits and flowers",
    "Ceramic mug with a crackled glaze in shades of turquoise and a whimsical polka dot design",
    "Glossy black mug with a neon geometric pattern",
    "Hand-thrown stoneware mug with a textured stripe pattern in earthy reds and browns",
    "Personalized photo mug with a panoramic beach sunset and a semi-glossy finish",
    "Clear glass mug with a frosted geometric pattern",
    "Wooden mug with a natural, unfinished surface and a pyrography (wood-burned) design",
    "Colorful ceramic mug with a hand-painted Mexican folk art motif",
    "Smooth porcelain mug with a watercolor wash effect in shades of blue",
    "Copper mug with a hammered texture and a simple engraved initial",
    "Glazed ceramic mug with a detailed map of a favorite city",
    "Matte stoneware mug with a speckled black and white glaze ",
    "Ceramic mug with a textured gradient design in ocean blues and greens",
    "Personalized ceramic mug with a birthstone-inspired watercolor design",
    "Vintage enamel mug with a faded nautical scene in shades of blue and white",
    "Textured ceramic mug with a woven pattern and a reactive glaze in shades of green and brown",
    "Smooth porcelain mug with a whimsical cartoon constellation map in navy blue and gold",
    "Hand-painted ceramic mug with a vibrant folk art design inspired by nature",
    "Matte stoneware mug with a speckled glaze in shades of gray and a bold abstract brushstroke",
    "Colorful ceramic mug with a playful cartoon food pattern in bright primary colors",
    "Personalized ceramic mug with a photorealistic portrait of a beloved pet and a glossy finish",
    "Earthenware mug with a rough, unglazed exterior and a smooth, glazed interior in a deep blue",
    "Vintage glass mug with a raised hobnail pattern and a soft amber tint",
    "Wooden mug carved from a single piece of burl wood, showcasing unique knots and swirls",
    "Colorful plastic mug with a soft-touch finish and a motivational quote in a cheerful font",
    "Hand-built ceramic mug with a pinched, organic shape and a textured glaze in earthy tones",
    "Smooth porcelain mug with a delicate floral decal in soft pastel colors",
    "Personalized ceramic mug with a full-color family photo and a matte finish",
    "Copper mug with a hammered finish and a quote engraved around the rim",
    "Ceramic mug with a speckled glaze in shades of cream and a playful handwritten recipe",
    "Textured stoneware mug with a geometric carving and a simple, unglazed finish",
    "Brightly colored enamel mug with a retro camping scene in a nostalgic color palette",
    "Personalized ceramic mug with a favorite fandom logo in bold, saturated colors",
    "Matte ceramic mug with a minimalist silhouette design in black and white",
    "Hand-thrown stoneware mug with a fingerprint texture and vibrant glaze drips",
    "Smooth glass mug with a frosted landscape design and subtle iridescent accents",
    "Ceramic mug with a woven basket texture and a warm, honey-colored glaze",
    "Vintage porcelain mug with a delicate gold filigree pattern and a soft pink interior",
    "Personalized ceramic mug with a watercolor galaxy design in shades of purple and blue",
    "Colorful ceramic mug with a raised polka dot pattern and a glossy glaze",
    "Stainless steel mug with a laser-etched geometric design and a brushed finish",
    "Glazed ceramic mug with a watercolor cityscape design in muted blues and grays",
    "Earthenware mug with a hand-painted abstract splatter design in bold, contrasting colors",
    "Smooth porcelain mug with a single Japanese kanji symbol in black calligraphy",
    "Clear glass mug with a sandblasted floral motif and a soft, diffused effect",
    "Colorful ceramic mug with a hand-painted ombré design in shades of sunset orange and pink",
    "Stoneware mug with a rough, textured exterior and a smooth, speckled glaze in cool blues",
    "Wooden mug with a natural bark edge and a simple, carved heart",
    "Personalized ceramic mug with a child's self-portrait in a colorful, whimsical style",
    "Glazed ceramic mug with a humorous optical illusion design",
    "Smooth porcelain mug with a photorealistic wildlife illustration",
    "Enamel mug with a classic sporting team logo and a glossy finish",
    "Ceramic mug with a textured honeycomb pattern and a vibrant yellow glaze",
    "Handmade stoneware mug with a textured swirl pattern and a deep, earthy red glaze",
    "Personalized ceramic mug with a hand-lettered name in an elegant script",
    "Matte ceramic mug with a minimalist pixel art design in a retro gaming style",
    "Vintage milk glass mug with a raised floral pattern and a soft, pearly white finish",
    "Glazed ceramic mug with a detailed botanical illustration of medicinal herbs",
    "Colorful plastic mug with a glitter-infused finish and a sparkly unicorn design",
    "Stoneware mug with a textured geometric carving and a simple, unglazed finish",
    "Hand-painted ceramic mug with a playful superhero comic book design",
    "Copper mug with a hammered texture and a personalized monogram engraving",
    "Glazed ceramic mug with a watercolor world map design in muted earth tones",
    "Personalized photo mug with a cherished memory and a vintage sepia tone",
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
for _ in range(5000):
    mug_description = random.choice(mug_descriptions)
    table_description = random.choice(table_descriptions)
    prompt = (
        "A "
        + mug_description
        + " on a "
        + table_description
        + ", "
        + random.choice(LIGHTINGS)
        + ", "
        + random.choice(STYLE_MEDIUM)
    )
    prompts.append(prompt)
# remove double commas
prompts = [prompt.replace(", ,", ",") for prompt in prompts]
# remove trailing comma at the end of the string
prompts = [prompt.rstrip(", ") for prompt in prompts]


# # clear if exists
# if target_directory.exists():
#     shutil.rmtree(target_directory)
# target_directory.mkdir(parents=True, exist_ok=True)


source_directory = DATA_DIR / "renders" / "mugs" / "objaverse-2500-tables"
target_directory = DATA_DIR / "diffusion_renders" / "mugs" / "run_10"

num_images_per_prompt = 4
num_prompts_per_scene = 2
diffusion_renderers = [
    (SD15RealisticCheckpointControlNetFromDepthRenderer, {"num_images_per_prompt": num_images_per_prompt}),
]

generate_diffusion_renders(
    source_directory, target_directory, diffusion_renderers, prompts, num_prompts_per_scene=num_prompts_per_scene
)
