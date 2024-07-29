import json

from dsd import DATA_DIR

# scenes
SHOE_SCENES_DIR = DATA_DIR / "scenes" / "shoes" / "gso-filtered-2500"
MUG_SCENES_DIR = DATA_DIR / "scenes" / "mugs" / "objaverse-filtered-2500-ral"
TSHIRT_SCENES_DIR = DATA_DIR / "scenes" / "tshirts" / "syncloth-filtered-2500-ordered"


SHOE_NO_TABLE_SCENES_DIR = DATA_DIR / "scenes" / "shoes" / "gso-filtered-2500-no-table"
MUG_NO_TABLE_SCENES_DIR = DATA_DIR / "scenes" / "mugs" / "objaverse-filtered-2500-ral-no-table"
TSHIRT_NO_TABLE_SCENES_DIR = DATA_DIR / "scenes" / "tshirts" / "syncloth-filtered-2500-no-table-ordered"

# real train  datasets
REAL_DATA_DIR = DATA_DIR / "real"
REAL_SHOES_TRAIN_DATASET = REAL_DATA_DIR / "shoes" / "dsd-shoes-robot-kpcentercropped" / "annotations_train.json"
REAL_MUGS_TRAIN_DATASET = REAL_DATA_DIR / "mugs" / "dsd-mugs-robot" / "annotations_train.json"
REAL_TSHIRTS_TRAIN_DATASET = REAL_DATA_DIR / "tshirts" / "tshirts-train.val-kpcentercropped" / "annotations_train.json"

# eval datasets
REAL_SHOES_VAL_DATASET = REAL_DATA_DIR / "shoes" / "dsd-shoes-robot-kpcentercropped" / "annotations_val.json"
REAL_SHOES_TEST_DATASET = REAL_DATA_DIR / "shoes" / "dsd-shoes-real_resized_512x512" / "annotations.json"


REAL_MUGS_VAL_DATASET = REAL_DATA_DIR / "mugs" / "dsd-mugs-robot" / "annotations_val.json"
REAL_MUGS_TEST_DATASET = REAL_DATA_DIR / "mugs" / "lab-mugs_resized_512x512" / "lab-mugs.json"

REAL_TSHIRTS_VAL_DATASET = REAL_DATA_DIR / "tshirts" / "tshirts-train.val-kpcentercropped" / "annotations_val.json"
REAL_TSHIRTS_TEST_DATASET = REAL_DATA_DIR / "tshirts" / "tshirts-test-kpcentercropped" / "annotations.json"

# random texture baselines
RANDOM_TEXTURE_BASELINE_DIR = DATA_DIR / "random_textures"
SHOE_RANDOM_TEXTURE_RENDER_DIR = RANDOM_TEXTURE_BASELINE_DIR / "shoes" / "001"
MUG_RANDOM_TEXTURE_RENDER_DIR = RANDOM_TEXTURE_BASELINE_DIR / "mugs" / "001"
TSHIRT_RANDOM_TEXTURE_RENDER_DIR = RANDOM_TEXTURE_BASELINE_DIR / "tshirts" / "001"

SHOE_RANDOM_TEXTURE_NO_TABLE_RENDER_DIR = RANDOM_TEXTURE_BASELINE_DIR / "shoes" / "no-table"
MUG_RANDOM_TEXTURE_NO_TABLE_RENDER_DIR = RANDOM_TEXTURE_BASELINE_DIR / "mugs" / "no-table"
TSHIRT_RANDOM_TEXTURE_NO_TABLE_RENDER_DIR = RANDOM_TEXTURE_BASELINE_DIR / "tshirts" / "no-table"

RANDOM_TEXTURE_BASELINE_SHOE_DATASET = DATA_DIR / "coco" / "shoes" / "random_textures" / "annotations.json"
RANDOM_TEXTURE_BASELINE_MUG_DATASET = DATA_DIR / "coco" / "mugs" / "random_textures" / "annotations.json"
RANDOM_TEXTURE_BASELINE_TSHIRT_DATASET = DATA_DIR / "coco" / "tshirts" / "random_textures" / "annotations.json"

RANDOM_TEXTURE_BASELINE_SHOE_NO_TABLE_DATASET = (
    DATA_DIR / "coco" / "shoes" / "random_textures_no_table" / "random_textures" / "annotations.json"
)
RANDOM_TEXTURE_BASELINE_MUG_NO_TABLE_DATASET = (
    DATA_DIR / "coco" / "mugs" / "random_textures_no_table" / "random_textures" / "annotations.json"
)
RANDOM_TEXTURE_BASELINE_TSHIRT_NO_TABLE_DATASET = (
    DATA_DIR / "coco" / "tshirts" / "random_textures_no_table" / "random_textures" / "annotations.json"
)

### DIFFUSION RENDERS

# prompt experiments
PROMPTS_CLASSNAME_SHOE_DATASET = (
    DATA_DIR
    / "coco"
    / "shoes"
    / "diffusion_renders"
    / "01-prompt-classname"
    / "ControlNetTXTFromDepthRenderer_ccs=1.5"
    / "annotations.json"
)
PROMPTS_CLASSNAME_MUG_DATASET = (
    DATA_DIR
    / "coco"
    / "mugs"
    / "diffusion_renders"
    / "01-prompt-classname"
    / "ControlNetTXTFromDepthRenderer_ccs=1.5"
    / "annotations.json"
)
PROMPTS_CLASSNAME_TSHIRT_DATASET = (
    DATA_DIR
    / "coco"
    / "tshirts"
    / "diffusion_renders"
    / "01-prompt-classname"
    / "ControlNetTXTFromDepthRenderer_ccs=1.5"
    / "annotations.json"
)

PROMPTS_GEMINI_SHOE_DATASET = (
    DATA_DIR
    / "coco"
    / "shoes"
    / "diffusion_renders"
    / "02-gemini-prompts"
    / "ControlNetTXTFromDepthRenderer_ccs=1.5"
    / "annotations.json"
)
PROMPTS_GEMINI_MUG_DATASET = (
    DATA_DIR
    / "coco"
    / "mugs"
    / "diffusion_renders"
    / "02-gemini-prompts"
    / "ControlNetTXTFromDepthRenderer_ccs=1.5"
    / "annotations.json"
)
PROMPTS_GEMINI_TSHIRT_DATASET = (
    DATA_DIR
    / "coco"
    / "tshirts"
    / "diffusion_renders"
    / "02-gemini-prompts"
    / "ControlNetTXTFromDepthRenderer_ccs=1.5"
    / "annotations.json"
)

PROMPTS_BLIP_SHOE_DATASET = (
    DATA_DIR
    / "coco"
    / "shoes"
    / "diffusion_renders"
    / "03-blip-captions"
    / "ControlNetTXTFromDepthRenderer_ccs=1.5"
    / "annotations.json"
)
PROMPTS_BLIP_MUG_DATASET = (
    DATA_DIR
    / "coco"
    / "mugs"
    / "diffusion_renders"
    / "03-blip-captions"
    / "ControlNetTXTFromDepthRenderer_ccs=1.5"
    / "annotations.json"
)
PROMPTS_BLIP_TSHIRT_DATASET = (
    DATA_DIR
    / "coco"
    / "tshirts"
    / "diffusion_renders"
    / "03-blip-captions"
    / "ControlNetTXTFromDepthRenderer_ccs=1.5"
    / "annotations.json"
)

# model experiments


# two-stage experiment
TWO_STAGE_BASELINE_TSHIRT_DATASET = (
    DATA_DIR
    / "coco"
    / "tshirts"
    / "diffusion_renders"
    / "04-two-stage-baseline"
    / "2stage:crop=Cropped:ControlNetTXTFromDepthRenderer_ccs=1.5,margin=10,only_change_mask=True,inp=SD2InpaintingRenderer_strength=1,dilation=1"
    / "annotations.json"
)
TWO_STAGE_BASELINE_MUG_DATASET = (
    DATA_DIR
    / "coco"
    / "mugs"
    / "diffusion_renders"
    / "04-two-stage-baseline"
    / "2stage:crop=Cropped:ControlNetTXTFromDepthRenderer_ccs=1.5,margin=10,only_change_mask=True,inp=SD2InpaintingRenderer_strength=1,dilation=1"
    / "annotations.json"
)
TWO_STAGE_BASELINE_SHOE_DATASET = (
    DATA_DIR
    / "coco"
    / "shoes"
    / "diffusion_renders"
    / "04-two-stage-baseline"
    / "2stage:crop=Cropped:ControlNetTXTFromDepthRenderer_ccs=1.5,margin=10,only_change_mask=True,inp=SD2InpaintingRenderer_strength=1,dilation=1"
    / "annotations.json"
)

ONE_STAGE_NO_TABLE_TSHIRT_DATASET = (
    DATA_DIR
    / "coco"
    / "tshirts"
    / "diffusion_renders"
    / "06-no-table-scenes"
    / "ControlNetTXTFromDepthRenderer_ccs=1.5"
    / "annotations.json"
)
ONE_STAGE_NO_TABLE_MUG_DATASET = (
    DATA_DIR
    / "coco"
    / "mugs"
    / "diffusion_renders"
    / "06-no-table-scenes"
    / "ControlNetTXTFromDepthRenderer_ccs=1.5"
    / "annotations.json"
)
ONE_STAGE_NO_TABLE_SHOE_DATASET = (
    DATA_DIR
    / "coco"
    / "shoes"
    / "diffusion_renders"
    / "06-no-table-scenes"
    / "ControlNetTXTFromDepthRenderer_ccs=1.5"
    / "annotations.json"
)

THREE_STAGE_TSHIRT_DATASET = (
    DATA_DIR / "coco" / "tshirts" / "diffusion_renders" / "07-three-stage" / "ThreeStageRenderer" / "annotations.json"
)
THREE_STAGE_MUG_DATASET = (
    DATA_DIR / "coco" / "mugs" / "diffusion_renders" / "07-three-stage" / "ThreeStageRenderer" / "annotations.json"
)
THREE_STAGE_SHOE_DATASET = (
    DATA_DIR / "coco" / "shoes" / "diffusion_renders" / "07-three-stage" / "ThreeStageRenderer" / "annotations.json"
)


DUAL_INPAINT_MUGS_DATASET = (
    DATA_DIR
    / "coco"
    / "mugs"
    / "diffusion_renders"
    / "08-dual-inpainting"
    / "DualInpaintRenderer"
    / "annotations.json"
)
DUAL_INPAINT_SHOES_DATASET = (
    DATA_DIR
    / "coco"
    / "shoes"
    / "diffusion_renders"
    / "08-dual-inpainting"
    / "DualInpaintRenderer"
    / "annotations.json"
)
DUAL_INPAINT_TSHIRTS_DATASET = (
    DATA_DIR
    / "coco"
    / "tshirts"
    / "diffusion_renders"
    / "08-dual-inpainting"
    / "DualInpaintRenderer"
    / "annotations.json"
)


dataset_paths = {k: str(v) for k, v in locals().items() if k.endswith("_DATASET")}

if __name__ == "__main__":
    for k, v in dataset_paths.items():
        try:
            num_images = len(json.load(open(v, "r"))["annotations"])
            print(f"{k}: # images =  {num_images}")
        except FileNotFoundError:
            print(f"{k}   path not found!")

    # dataset_dict = {
    #     "shoes": {
    #         "train": REAL_SHOES_TRAIN_DATASET,
    #         "val": REAL_SHOES_VAL_DATASET,
    #         "test": REAL_SHOES_TEST_DATASET,
    #     },
    #     "mugs": {
    #         "train": REAL_MUGS_TRAIN_DATASET,
    #         "val": REAL_MUGS_VAL_DATASET,
    #         "test": REAL_MUGS_TEST_DATASET,
    #     },
    #     "tshirts": {
    #         "train": REAL_TSHIRTS_TRAIN_DATASET,
    #         "val": REAL_TSHIRTS_VAL_DATASET,
    #         "test": REAL_TSHIRTS_TEST_DATASET,
    #     },
    # }

    # for k, v in dataset_paths.items():
    #     print(k)
    #     for k2, v2 in v.items():
    #         # open json file
    #         with open(v2, "r") as f:
    #             data = json.load(f)
    #             num_images = len(data["images"])
    #             print(f"{k}-{k2}    num images: {num_images}")
