import subprocess

from experiments.train_keypoints import MUG_DICT, SHOE_DICT, TSHIRT_DICT, _create_command
from paths import (  # noqa
    CCS_COMPARISON_05_MUG_DATASET,
    CCS_COMPARISON_05_SHOE_DATASET,
    CCS_COMPARISON_05_TSHIRT_DATASET,
    CCS_COMPARISON_10_MUG_DATASET,
    CCS_COMPARISON_10_SHOE_DATASET,
    CCS_COMPARISON_10_TSHIRT_DATASET,
    CCS_COMPARISON_15_MUG_DATASET,
    CCS_COMPARISON_15_SHOE_DATASET,
    CCS_COMPARISON_15_TSHIRT_DATASET,
    CCS_COMPARISON_20_MUG_DATASET,
    CCS_COMPARISON_20_SHOE_DATASET,
    CCS_COMPARISON_20_TSHIRT_DATASET,
    CCS_COMPARISON_25_MUG_DATASET,
    CCS_COMPARISON_25_SHOE_DATASET,
    CCS_COMPARISON_25_TSHIRT_DATASET,
    DUAL_INPAINT_MUGS_DATASET,
    DUAL_INPAINT_SHOES_DATASET,
    DUAL_INPAINT_TSHIRTS_DATASET,
    IMG2IMG_MUGS_DATASET,
    IMG2IMG_SHOES_DATASET,
    IMG2IMG_TSHIRTS_DATASET,
    ONE_STAGE_LARGE_MUG_DATASET,
    ONE_STAGE_LARGE_SHOE_DATASET,
    ONE_STAGE_LARGE_TSHIRT_DATASET,
    ONE_STAGE_NO_TABLE_MUG_DATASET,
    ONE_STAGE_NO_TABLE_SHOE_DATASET,
    ONE_STAGE_NO_TABLE_TSHIRT_DATASET,
    PROMPTS_BLIP_MUG_DATASET,
    PROMPTS_BLIP_SHOE_DATASET,
    PROMPTS_BLIP_TSHIRT_DATASET,
    PROMPTS_CLASSNAME_MUG_DATASET,
    PROMPTS_CLASSNAME_SHOE_DATASET,
    PROMPTS_CLASSNAME_TSHIRT_DATASET,
    PROMPTS_GEMINI_MUG_DATASET,
    PROMPTS_GEMINI_SHOE_DATASET,
    PROMPTS_GEMINI_TSHIRT_DATASET,
    RANDOM_TEXTURE_MUG_LARGE_DATASET,
    RANDOM_TEXTURE_SHOE_LARGE_DATASET,
    RANDOM_TEXTURE_TSHIRT_LARGE_DATASET,
    THREE_STAGE_MUG_DATASET,
    THREE_STAGE_SHOE_DATASET,
    THREE_STAGE_TSHIRT_DATASET,
    TWO_STAGE_BASELINE_MUG_DATASET,
    TWO_STAGE_BASELINE_SHOE_DATASET,
    TWO_STAGE_BASELINE_TSHIRT_DATASET,
    TWO_STAGE_LARGER_MASK_MUGS_DATASET,
    TWO_STAGE_LARGER_MASK_SHOES_DATASET,
    TWO_STAGE_LARGER_MASK_TSHIRTS_DATASET,
)


def train_on_prompts_classname():
    epochs = 20

    # mugs
    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-prompts-classname"
    mug_dict["json_dataset_path"] = str(PROMPTS_CLASSNAME_MUG_DATASET)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    # subprocess.run(command, shell=True)

    # shoes
    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-prompts-classname"
    shoe_dict["json_dataset_path"] = str(PROMPTS_CLASSNAME_SHOE_DATASET)
    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    # subprocess.run(command, shell=True)

    # tshirts
    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-prompts-classname"
    tshirt_dict["json_dataset_path"] = str(PROMPTS_CLASSNAME_TSHIRT_DATASET)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)


def train_on_prompts_gemini():
    epochs = 20

    # mugs
    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-prompts-gemini"
    mug_dict["json_dataset_path"] = str(PROMPTS_GEMINI_MUG_DATASET)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    # subprocess.run(command, shell=True)

    # shoes
    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-prompts-gemini"
    shoe_dict["json_dataset_path"] = str(PROMPTS_GEMINI_SHOE_DATASET)
    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    # subprocess.run(command, shell=True)

    # tshirts
    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-prompts-gemini"
    tshirt_dict["json_dataset_path"] = str(PROMPTS_GEMINI_TSHIRT_DATASET)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)


def train_on_prompts_blip():
    epochs = 20

    # mugs
    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-prompts-blip"
    mug_dict["json_dataset_path"] = str(PROMPTS_BLIP_MUG_DATASET)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    subprocess.run(command, shell=True)

    # shoes
    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-prompts-blip"
    shoe_dict["json_dataset_path"] = str(PROMPTS_BLIP_SHOE_DATASET)
    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    # tshirts
    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-prompts-blip"
    tshirt_dict["json_dataset_path"] = str(PROMPTS_BLIP_TSHIRT_DATASET)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)


def train_on_2_stage_baseline():
    epochs = 20
    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-2-stage-baseline"
    mug_dict["json_dataset_path"] = str(TWO_STAGE_BASELINE_MUG_DATASET)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    print(command)
    subprocess.run(command, shell=True)

    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-2-stage-baseline"
    shoe_dict["json_dataset_path"] = str(TWO_STAGE_BASELINE_SHOE_DATASET)
    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-2-stage-baseline"
    tshirt_dict["json_dataset_path"] = str(TWO_STAGE_BASELINE_TSHIRT_DATASET)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)


def train_on_no_table():
    epochs = 20
    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "1-stage-mugs-no-table"
    mug_dict["json_dataset_path"] = str(ONE_STAGE_NO_TABLE_MUG_DATASET)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    print(command)
    subprocess.run(command, shell=True)

    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "1-stage-shoes-no-table"
    shoe_dict["json_dataset_path"] = str(ONE_STAGE_NO_TABLE_SHOE_DATASET)
    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "1-stage-tshirts-no-table"
    tshirt_dict["json_dataset_path"] = str(ONE_STAGE_NO_TABLE_TSHIRT_DATASET)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)


def train_on_three_stage():
    epochs = 20
    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-3-stage"
    mug_dict["json_dataset_path"] = str(THREE_STAGE_MUG_DATASET)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    print(command)
    subprocess.run(command, shell=True)

    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-3-stage"
    shoe_dict["json_dataset_path"] = str(THREE_STAGE_SHOE_DATASET)
    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-3-stage"
    tshirt_dict["json_dataset_path"] = str(THREE_STAGE_TSHIRT_DATASET)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)


def train_on_dual_inpainting_diffusion():
    epochs = 20
    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-dual-inpainting"
    mug_dict["json_dataset_path"] = str(DUAL_INPAINT_MUGS_DATASET)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    print(command)
    subprocess.run(command, shell=True)

    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-dual-inpainting"
    shoe_dict["json_dataset_path"] = str(DUAL_INPAINT_SHOES_DATASET)
    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-dual-inpainting"
    tshirt_dict["json_dataset_path"] = str(DUAL_INPAINT_TSHIRTS_DATASET)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)


def train_on_img2img_diffusion():
    epochs = 20
    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-img2img"
    mug_dict["json_dataset_path"] = str(IMG2IMG_MUGS_DATASET)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    print(command)
    subprocess.run(command, shell=True)

    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-img2img"
    shoe_dict["json_dataset_path"] = str(IMG2IMG_SHOES_DATASET)
    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-img2img"
    tshirt_dict["json_dataset_path"] = str(IMG2IMG_TSHIRTS_DATASET)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)


def train_on_2_stage_larger_mask():
    epochs = 20
    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-2-stage-larger-mask"
    mug_dict["json_dataset_path"] = str(TWO_STAGE_LARGER_MASK_MUGS_DATASET)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    print(command)
    subprocess.run(command, shell=True)

    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-2-stage-larger-mask"
    shoe_dict["json_dataset_path"] = str(TWO_STAGE_LARGER_MASK_SHOES_DATASET)
    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-2-stage-larger-mask"
    tshirt_dict["json_dataset_path"] = str(TWO_STAGE_LARGER_MASK_TSHIRTS_DATASET)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)


def train_on_ccs_comparison():
    # 0.5
    epochs = 20

    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-ccs-0.5"
    mug_dict["json_dataset_path"] = str(CCS_COMPARISON_05_MUG_DATASET)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    subprocess.run(command, shell=True)

    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-ccs-0.5"
    shoe_dict["json_dataset_path"] = str(CCS_COMPARISON_05_SHOE_DATASET)
    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-ccs-0.5"
    tshirt_dict["json_dataset_path"] = str(CCS_COMPARISON_05_TSHIRT_DATASET)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)

    # 1.0
    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-ccs-1.0"
    mug_dict["json_dataset_path"] = str(CCS_COMPARISON_10_MUG_DATASET)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    subprocess.run(command, shell=True)

    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-ccs-1.0"
    shoe_dict["json_dataset_path"] = str(CCS_COMPARISON_10_SHOE_DATASET)
    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-ccs-1.0"
    tshirt_dict["json_dataset_path"] = str(CCS_COMPARISON_10_TSHIRT_DATASET)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)

    # 1.5
    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-ccs-1.5"
    mug_dict["json_dataset_path"] = str(CCS_COMPARISON_15_MUG_DATASET)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    subprocess.run(command, shell=True)

    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-ccs-1.5"
    shoe_dict["json_dataset_path"] = str(CCS_COMPARISON_15_SHOE_DATASET)
    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-ccs-1.5"
    tshirt_dict["json_dataset_path"] = str(CCS_COMPARISON_15_TSHIRT_DATASET)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)

    # 2.0
    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-ccs-2.0"
    mug_dict["json_dataset_path"] = str(CCS_COMPARISON_20_MUG_DATASET)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    subprocess.run(command, shell=True)

    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-ccs-2.0"
    shoe_dict["json_dataset_path"] = str(CCS_COMPARISON_20_SHOE_DATASET)
    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-ccs-2.0"
    tshirt_dict["json_dataset_path"] = str(CCS_COMPARISON_20_TSHIRT_DATASET)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)

    # 2.5
    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-ccs-2.5"
    mug_dict["json_dataset_path"] = str(CCS_COMPARISON_25_MUG_DATASET)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    subprocess.run(command, shell=True)

    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-ccs-2.5"
    shoe_dict["json_dataset_path"] = str(CCS_COMPARISON_25_SHOE_DATASET)
    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-ccs-2.5"
    tshirt_dict["json_dataset_path"] = str(CCS_COMPARISON_25_TSHIRT_DATASET)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)


def train_on_scale_experiment_one_stage_diffusion():
    import pathlib

    def dataset_to_split_name(dataset, size):
        return f"{str(pathlib.Path(dataset).with_suffix(''))}_{size}.json"

    epochs = 20

    # 1K
    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-scale-diffusion-1K"
    print(dataset_to_split_name(ONE_STAGE_LARGE_MUG_DATASET, 1000))
    mug_dict["json_dataset_path"] = dataset_to_split_name(ONE_STAGE_LARGE_MUG_DATASET, 1000)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    subprocess.run(command, shell=True)

    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-scale-diffusion-1K"
    shoe_dict["json_dataset_path"] = dataset_to_split_name(ONE_STAGE_LARGE_SHOE_DATASET, 1000)
    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-scale-diffusion-1K"
    tshirt_dict["json_dataset_path"] = dataset_to_split_name(ONE_STAGE_LARGE_TSHIRT_DATASET, 1000)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)

    # 2K
    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-scale-diffusion-2K"
    mug_dict["json_dataset_path"] = dataset_to_split_name(ONE_STAGE_LARGE_MUG_DATASET, 2000)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    subprocess.run(command, shell=True)

    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-scale-diffusion-2K"
    shoe_dict["json_dataset_path"] = dataset_to_split_name(ONE_STAGE_LARGE_SHOE_DATASET, 2000)
    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-scale-diffusion-2K"
    tshirt_dict["json_dataset_path"] = dataset_to_split_name(ONE_STAGE_LARGE_TSHIRT_DATASET, 2000)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)

    # 5K

    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-scale-diffusion-5K"
    mug_dict["json_dataset_path"] = dataset_to_split_name(ONE_STAGE_LARGE_MUG_DATASET, 5000)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    subprocess.run(command, shell=True)

    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-scale-diffusion-5K"
    shoe_dict["json_dataset_path"] = dataset_to_split_name(ONE_STAGE_LARGE_SHOE_DATASET, 5000)
    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-scale-diffusion-5K"
    tshirt_dict["json_dataset_path"] = dataset_to_split_name(ONE_STAGE_LARGE_TSHIRT_DATASET, 5000)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)

    # 10K

    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-scale-diffusion-10K"
    mug_dict["json_dataset_path"] = dataset_to_split_name(ONE_STAGE_LARGE_MUG_DATASET, 10000)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    subprocess.run(command, shell=True)

    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-scale-diffusion-10K"
    shoe_dict["json_dataset_path"] = dataset_to_split_name(ONE_STAGE_LARGE_SHOE_DATASET, 10000)
    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-scale-diffusion-10K"
    tshirt_dict["json_dataset_path"] = dataset_to_split_name(ONE_STAGE_LARGE_TSHIRT_DATASET, 10000)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)


def train_on_scale_experiment_random_textures():
    import pathlib

    def dataset_to_split_name(dataset, size):
        return f"{str(pathlib.Path(dataset).with_suffix(''))}_{size}.json"

    epochs = 20

    # 1K
    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-scale-random-textures-1K"
    mug_dict["json_dataset_path"] = dataset_to_split_name(RANDOM_TEXTURE_MUG_LARGE_DATASET, 1000)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    subprocess.run(command, shell=True)

    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-scale-random-textures-1K"
    shoe_dict["json_dataset_path"] = dataset_to_split_name(RANDOM_TEXTURE_SHOE_LARGE_DATASET, 1000)
    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-scale-random-textures-1K"
    tshirt_dict["json_dataset_path"] = dataset_to_split_name(RANDOM_TEXTURE_TSHIRT_LARGE_DATASET, 1000)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)

    # 2K
    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-scale-random-textures-2K"
    mug_dict["json_dataset_path"] = dataset_to_split_name(RANDOM_TEXTURE_MUG_LARGE_DATASET, 2000)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    subprocess.run(command, shell=True)

    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-scale-random-textures-2K"
    shoe_dict["json_dataset_path"] = dataset_to_split_name(RANDOM_TEXTURE_SHOE_LARGE_DATASET, 2000)
    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-scale-random-textures-2K"
    tshirt_dict["json_dataset_path"] = dataset_to_split_name(RANDOM_TEXTURE_TSHIRT_LARGE_DATASET, 2000)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)

    # 5K

    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-scale-random-textures-5K"
    mug_dict["json_dataset_path"] = dataset_to_split_name(RANDOM_TEXTURE_MUG_LARGE_DATASET, 5000)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    subprocess.run(command, shell=True)

    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-scale-random-textures-5K"
    shoe_dict["json_dataset_path"] = dataset_to_split_name(RANDOM_TEXTURE_SHOE_LARGE_DATASET, 5000)
    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-scale-random-textures-5K"
    tshirt_dict["json_dataset_path"] = dataset_to_split_name(RANDOM_TEXTURE_TSHIRT_LARGE_DATASET, 5000)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)

    # 10K

    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-scale-random-textures-10K"
    mug_dict["json_dataset_path"] = dataset_to_split_name(RANDOM_TEXTURE_MUG_LARGE_DATASET, 10000)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    subprocess.run(command, shell=True)

    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-scale-random-textures-10K"
    shoe_dict["json_dataset_path"] = dataset_to_split_name(RANDOM_TEXTURE_SHOE_LARGE_DATASET, 10000)
    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-scale-random-textures-10K"
    tshirt_dict["json_dataset_path"] = dataset_to_split_name(RANDOM_TEXTURE_TSHIRT_LARGE_DATASET, 10000)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    # train_on_prompts_blip()
    # train_on_prompts_gemini()
    # train_on_prompts_classname()
    # train_on_2_stage_baseline()
    # train_on_no_table()
    # train_on_three_stage()
    # train_on_dual_inpainting_diffusion()
    # train_on_img2img_diffusion()
    # train_on_2_stage_larger_mask()
    # train_on_ccs_comparison()
    # train_on_scale_experiment_one_stage_diffusion()
    train_on_scale_experiment_random_textures()
