"""
Geminim promtps for description of mugs, shoes, tshirts and surfaces

table: https://g.co/gemini/share/05bd603583fd
mug: https://g.co/gemini/share/b70053a43ece
shoe: https://g.co/gemini/share/42f627975594
tshirt: https://g.co/gemini/share/035391db88be

"""

import random

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


shoe_descriptions = [
    "Crimson red patent leather pumps with a stiletto heel",
    "Distressed brown leather Chelsea boots with elastic side panels",
    "Cream-colored canvas espadrilles with woven jute soles",
    "Cobalt blue suede loafers with a gold horsebit detail",
    "Emerald green satin ballet flats with a dainty bow",
    "Sunflower yellow woven slides with criss-cross straps",
    "Deep purple velvet ankle boots with a chunky block heel",
    "Metallic silver platform sandals with a cork wedge heel",
    "Coral pink mesh sneakers with neon orange accents",
    "Leopard print calf-hair pumps with a pointed toe",
    "Black and white striped canvas sneakers with high tops",
    "Burgundy leather wingtip brogues with perforated detailing",
    "Tan suede desert boots with crepe rubber soles",
    "Lavender satin mules with a pearl-encrusted heel",
    "Olive green canvas hiking boots with metal D-rings",
    "Fuchsia pink faux fur slides with a fluffy strap",
    "Navy blue suede boat shoes with rawhide laces",
    "Turquoise beaded sandals with a delicate ankle strap",
    "White leather sneakers with a chunky platform sole",
    "Gold metallic leather sandals with a thin stiletto heel",
    "Rainbow tie-dye canvas slip-on sneakers",
    "Python print leather ankle boots with a side zipper",
    "Denim fabric espadrilles with a frayed edge",
    "Crocodile embossed leather loafers with a tassel",
    "Checkered canvas Vans-style slip-on shoes",
    "Peach-colored suede mules with a knotted detail",
    "Charcoal gray knit sock sneakers with a rubber sole",
    "Rust-colored leather Chelsea boots with a Cuban heel",
    "Mint green canvas espadrilles with a striped sole",
    "Ombre effect leather pumps fading from blue to purple",
    "Silver glitter platform sandals with a chunky heel",
    "Snakeskin print leather mules with an open toe",
    "Beige suede desert boots with fringe detailing",
    "Hot pink satin ballet flats with a rhinestone buckle",
    "Neon yellow mesh running shoes with reflective accents",
    "Camouflage print canvas sneakers with a high top",
    "Chocolate brown leather brogues with a wingtip design",
    "Rose gold metallic leather pumps with a pointed toe",
    "Floral print canvas espadrilles with a braided jute sole",
    "Teal blue suede loafers with a penny keeper detail",
    "Acid wash denim fabric sneakers with a lace-up closure",
    "Metallic bronze leather sandals with a gladiator style",
    "Polka dot print canvas slip-on shoes with a rounded toe",
    "Burgundy velvet ankle boots with a side zipper and buckle",
    "Cream-colored crochet sandals with a wooden bead accent",
    "Orange and blue colorblock mesh running shoes",
    "Black leather combat boots with a lace-up front and buckle",
    "White leather mules with a braided strap and open toe",
    "Tan suede Chelsea boots with a pointed toe and elastic gussets",
    "Neon green jelly sandals with a chunky translucent strap",
    "Distressed brown leather work boots with a steel toe cap",
    "White leather sneakers with a perforated toe box and gum sole",
    "Black canvas slip-on shoes with a minimalist design",
    "Navy blue suede boat shoes with contrasting white stitching",
    "Tan leather sandals with adjustable straps and a cork footbed",
    "Gray mesh running shoes with a cushioned midsole",
    "Brown suede Chelsea boots with a comfortable elastic gusset",
    "Black leather Oxfords with a brogue pattern and a low heel",
    "White canvas sneakers with a classic lace-up design",
    "Brown leather loafers with a horsebit detail and a rubber sole",
    "Black suede pumps with a stiletto heel and a pointed toe",
    "Nude patent leather pumps with a slingback strap",
    "Red leather ankle boots with a block heel and a side zipper",
    "Gray suede ankle boots with a stacked heel and a lace-up front",
    "Brown leather wingtip brogues with a medallion toe",
    "Black patent leather loafers with a tassel and a penny keeper",
    "White leather sneakers with a platform sole and a velcro closure",
    "Black leather sandals with a T-strap and a buckle closure",
    "Brown leather sandals with a criss-cross design and a flat sole",
    "Beige suede sandals with a knotted detail and a wedge heel",
    "Gray knit sock sneakers with a pull-on design and a rubber sole",
    "Black leather combat boots with a lace-up front and a side zipper",
    "Brown leather hiking boots with a lug sole and a waterproof membrane",
    "Green canvas hiking boots with a padded collar and a breathable lining",
    "Blue denim fabric sneakers with a lace-up closure and a contrast stitching",
    "White leather sneakers with a minimalist design and a perforated logo",
    "Black leather sandals with a thong strap and a cushioned footbed",
    "Brown leather sandals with a slingback strap and a braided detail",
    "Tan leather sandals with a toe loop and a contoured footbed",
    "Gray suede sandals with a criss-cross design and a wedge heel",
    "Black leather mules with a pointed toe and a kitten heel",
    "White leather mules with an open back and a block heel",
    "Brown leather clogs with a wooden sole and a metal buckle",
    "Black suede clogs with a closed back and a studded detail",
    "Tan leather slides with a single band and a cushioned footbed",
    "White leather slides with a double band and a logo detail",
    "Brown leather slippers with a shearling lining and a rubber sole",
    "Gray felt slippers with a pom-pom detail and a non-slip sole",
    "Black velvet slippers with an embroidered design and a leather sole",
    "Red satin ballet flats with a bow detail and a flexible sole",
]

tshirt_descriptions = [
    "Vintage black cotton tee with a faded band logo",
    "Bright pink tie-dye crop top with raw edges",
    "Forest green linen T-shirt with embroidered leaves",
    "Navy blue striped boatneck tee with three-quarter sleeves",
    "Cream-colored silk V-neck with delicate lace trim",
    "Heathered gray oversized T-shirt with rolled cuffs",
    "Olive green muscle tee with a distressed graphic print",
    "Coral burnout velvet tee with a relaxed fit",
    "Sunny yellow linen blend T-shirt with a front pocket",
    "Marbled turquoise and white jersey tee with a scoop neck",
    "Charcoal gray waffle knit henley with wooden buttons",
    "Lavender organic cotton T-shirt with cap sleeves",
    "Dusty rose slub knit tee with a raw hem",
    "Burnt orange thermal tee with a ribbed crew neck",
    "Mint green bamboo blend T-shirt with a high neckline",
    "Black and white gingham pocket tee with contrasting buttons",
    "Deep red modal T-shirt with a cowl neck",
    "Sky blue jersey tee with a vintage sports logo",
    "White ribbed tank top with a racerback",
    "Neon orange mesh tee with a cropped silhouette",
    "Silver metallic foil-printed T-shirt with a boxy fit",
    "Indigo blue chambray button-down shirt with a relaxed fit",
    "Emerald green linen blend T-shirt with a ruffled hem",
    "Mustard yellow tie-front crop top with flutter sleeves",
    "Royal blue cotton polo shirt with embroidered crest",
    "Burgundy thermal henley with a waffle knit pattern",
    "Plum purple satin tee with a draped neckline",
    "Sunflower yellow linen T-shirt with a scoop neck",
    "Teal blue jersey tee with a vintage band logo",
    "Peachy pink ribbed tank top with spaghetti straps",
    "Magenta tie-dye crop top with a knot detail",
    "Chocolate brown organic cotton tee with a pocket",
    "Aqua blue linen blend T-shirt with cap sleeves",
    "Gold metallic thread embroidered T-shirt with a V-neck",
    "Crimson red modal tee with a ruffled hem",
    "Seafoam green jersey tee with a nautical stripe",
    "Ivory white silk T-shirt with a pleated front",
    "Tan suede fringe trim T-shirt with a relaxed fit",
    "Denim blue chambray button-down shirt with rolled sleeves",
    "Cobalt blue cotton polo shirt with contrasting collar",
    "A vibrant crimson crew neck tee, made from soft cotton with a subtle ribbed texture.",
    "A classic white v-neck tee, featuring a bold black and white striped pattern.",
    "A relaxed-fit heather gray tee, crafted from a breathable linen-cotton blend.",
    "A sunny yellow scoop neck tee, embellished with delicate floral embroidery.",
    "A trendy olive green oversized tee, adorned with a quirky animal print.",
    "A deep navy blue polo shirt, tailored from a luxurious pique cotton fabric.",
    "A vintage-inspired dusty rose ringer tee, boasting contrasting trim details.",
    "A playful coral raglan tee, designed with color-blocked sleeves and a vintage graphic.",
    "A soft lavender Henley tee, featuring a buttoned placket and a relaxed silhouette.",
    "A sophisticated charcoal tie-dye tee, showcasing a mesmerizing swirl pattern.",
    "A vibrant turquoise tank top, made from a lightweight jersey with a racerback design.",
    "A crisp white button-down shirt, crafted from a breathable linen fabric with a subtle sheen.",
    "A bold red and blue plaid flannel shirt, perfect for layering over a basic tee.",
    "A rustic brown chambray shirt, offering a versatile and stylish option for casual wear.",
    "A sleek black turtleneck, made from a soft and stretchy merino wool blend.",
    "A cozy cream cable knit sweater, ideal for colder weather and layering over shirts.",
    "A relaxed-fit beige hoodie, featuring a kangaroo pocket and a drawstring hood.",
    "A sporty navy blue zip-up jacket, made from a water-resistant technical fabric.",
    "A stylish olive green bomber jacket, crafted from a supple leather with a ribbed collar.",
    "A classic denim jacket in a medium wash, perfect for pairing with jeans and a tee.",
    "A trendy black leather jacket with silver hardware, adding a touch of edge to any outfit.",
    "A minimalist white linen blazer, ideal for layering over a dress or a blouse for a polished look.",
    "A vibrant yellow raincoat, made from a waterproof material with a hood and zipper closure.",
    "A cozy gray wool scarf, perfect for adding warmth and style to any winter ensemble.",
    "A stylish brown leather belt with a silver buckle, complementing any outfit from casual to formal.",
    "A pair of classic black sunglasses with polarized lenses, offering protection and style.",
    "A set of colorful printed socks, adding a fun and quirky touch to any footwear choice.",
    "A minimalist silver pendant necklace, adding a subtle touch of elegance to any neckline.",
    "A pair of delicate gold hoop earrings, perfect for everyday wear and special occasions.",
    "A vibrant printed silk scarf, adding a touch of luxury and color to any outfit.",
    "A pair of comfortable leather loafers, perfect for both casual and formal settings.",
    "A versatile black tote bag, ideal for carrying essentials and adding a touch of sophistication.",
    "A playful printed canvas tote bag, perfect for carrying groceries or everyday items.",
    "A stylish leather backpack, ideal for travel or everyday use, combining fashion and functionality.",
    "A sporty nylon backpack with multiple compartments, perfect for carrying gear and staying organized.",
    "A cozy fleece blanket in a neutral tone, perfect for snuggling up on the couch or adding warmth to a bed.",
    "A set of colorful decorative throw pillows, adding a pop of personality to any living space.",
    "A minimalist ceramic vase, perfect for displaying fresh flowers or adding a touch of elegance to a room.",
    "A set of rustic wooden coasters, protecting surfaces and adding a touch of natural charm.",
    "A playful printed shower curtain, brightening up any bathroom and adding a touch of fun.",
    "A set of soft and absorbent cotton towels in a neutral color, perfect for everyday use.",
    "A luxurious silk eye mask, promoting restful sleep and adding a touch of indulgence to bedtime.",
    "A pair of cozy fleece slippers, keeping feet warm and comfortable around the house.",
    "A set of colorful reusable straws, reducing waste and adding a touch of fun to any beverage.",
    "A durable and stylish water bottle, encouraging hydration and reducing single-use plastic waste.",
    "A versatile canvas tote bag with a printed design, perfect for carrying groceries or everyday essentials.",
    "A set of reusable beeswax food wraps, providing an eco-friendly alternative to plastic wrap for storing food.",
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
    "round wooden table with a natural finish",
    "oval glass table with a mirrored base",
    "square metal table with a distressed finish",
    "rectangular concrete table with a polished surface",
    "triangular acrylic table with a translucent effect",
    "wrought-iron patio table with a floral design",
    "adjustable-height standing desk with a bamboo top",
    "reclaimed wood table with a patchwork pattern",
    "live-edge wood slab table with a unique shape",
    "industrial-style table with exposed gears and bolts",
    "mid-century modern table with hairpin legs",
    "wicker table with a woven glass top",
    "terrazzo table with colorful chips embedded",
    "tufted ottoman that doubles as a coffee table",
    "folding table with a faux marble top",
    "painted wooden table with a folk art design",
    "vintage sewing machine base repurposed as a table",
    "marble-top bistro table with a cast iron base",
    "antique writing desk with leather inlay",
    "hexagonal table with a honeycomb pattern",
    "oversized picnic table with benches attached",
    "compact folding table for camping trips",
    "table with a built-in wine rack",
    "table with a hidden compartment for games",
    "table with a rotating top for easy sharing",
    "table with a built-in cooler for drinks",
    "table with a hidden drawer for storage",
    "table with a chalkboard surface for doodling",
    "table with a built-in fire pit for warmth",
    "table with a built-in planter for herbs",
    "table with a built-in chessboard",
    "table with a built-in backgammon board",
    "table with a built-in checkers board",
    "table with a built-in Jenga set",
    "table with a built-in dominoes set",
    "table with a built-in playing card holder",
    "table with a built-in dice tray",
    "table with a built-in poker chip tray",
    "table with a built-in bottle opener",
    "table with a built-in corkscrew",
    "table with a built-in wine glass holder",
    "table with a built-in beer mug holder",
    "table with a built-in candle holder",
    "table with a built-in vase for flowers",
    "table with a built-in photo frame",
    "table with a built-in music player",
    "table with an integrated chessboard and drawers for pieces",
    "table featuring a lazy Susan for serving appetizers",
    "table with a hidden cooler compartment for beverages",
    "table with a removable top revealing hidden storage",
    "table showcasing an integrated wine rack and stemware holders",
    "table with a detachable grill top for outdoor cooking",
    "table equipped with an ice bucket for chilling drinks",
    "table featuring a removable cutting board for food prep",
    "table designed with an herb garden planter",
    "table with a built-in fish tank",
    "table with an attached bird feeder",
    "table incorporating a bird bath",
    "table with a designated space for a dog bowl",
    "table offering a cozy cat bed",
    "table with a convenient book holder",
    "table displaying a stylish magazine rack",
    "table showcasing a newspaper holder",
    "table designed with a laptop stand",
    "table featuring a dedicated tablet holder",
    "table with a wireless smartphone charging pad",
    "table with an integrated speaker system",
    "table designed to accommodate a projector",
    "table featuring a TV screen mount",
    "table with a fireplace insert",
    "table with a space heater attachment",
    "table with an air conditioning unit",
    "table with a fan attachment",
    "table with a humidifier feature",
    "table with a dehumidifier function",
    "table with an air purifier",
    "table with an aquarium filter",
    "table with a bird feeder camera",
    "table with a bird bath fountain",
    "table with a dog bowl water dispenser",
    "table with a heated cat bed",
    "table with a built-in book light",
    "table with an attached magnifying glass",
    "table featuring a telescope mount",
    "table with a microscope stand",
    "table with a camera mount",
    "table with a microphone stand",
    "table designed with a keyboard tray",
    "table with a dedicated mouse pad",
    "table featuring an integrated trackpad",
]

LIGHTINGS = ["", "ambient light", "studio lighting", "natural light"]
STYLE_MEDIUM = ["", "RAW", "Photorealistic", "Photography, 4K"]


random.seed(2024)
tshirt_prompts = []
mug_prompts = []
shoe_prompts = []
for _ in range(5000):
    tshirt_prompts.append((random.choice(tshirt_descriptions), random.choice(table_descriptions)))
    mug_prompts.append((random.choice(mug_descriptions), random.choice(table_descriptions)))
    shoe_prompts.append((random.choice(shoe_descriptions), random.choice(table_descriptions)))

# store the prompts as txt files
with open("mug_prompts.txt", "w") as f:
    for prompt in mug_prompts:
        f.write(f"{prompt[0]};{prompt[1]}\n")

with open("tshirt_prompts.txt", "w") as f:
    for prompt in tshirt_prompts:
        f.write(f"{prompt[0]};{prompt[1]}\n")

with open("shoe_prompts.txt", "w") as f:
    for prompt in shoe_prompts:
        f.write(f"{prompt[0]};{prompt[1]}\n")
