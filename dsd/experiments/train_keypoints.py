import subprocess

COMMAND = "keypoint-detection train  --augment_train"
from paths import (  # noqa
    RANDOM_TEXTURE_BASELINE_MUG_DATASET,
    RANDOM_TEXTURE_BASELINE_MUG_NO_TABLE_DATASET,
    RANDOM_TEXTURE_BASELINE_SHOE_DATASET,
    RANDOM_TEXTURE_BASELINE_SHOE_NO_TABLE_DATASET,
    RANDOM_TEXTURE_BASELINE_TSHIRT_DATASET,
    RANDOM_TEXTURE_BASELINE_TSHIRT_NO_TABLE_DATASET,
    REAL_MUGS_TEST_DATASET,
    REAL_MUGS_TRAIN_DATASET,
    REAL_MUGS_VAL_DATASET,
    REAL_SHOES_TEST_DATASET,
    REAL_SHOES_TRAIN_DATASET,
    REAL_SHOES_VAL_DATASET,
    REAL_TSHIRTS_TEST_DATASET,
    REAL_TSHIRTS_TRAIN_DATASET,
    REAL_TSHIRTS_VAL_DATASET,
)

DEFAULT_DICT = {
    "keypoint_channel_configuration": None,
    "accelerator": "gpu",
    "ap_epoch_freq": 1,
    # "check_val_every_n_epoch": 10,
    "backbone_type": "MaxVitUnet",
    "devices": 1,
    "early_stopping_relative_threshold": -1,
    "json_dataset_path": "",
    "json_test_dataset_path": "",
    "json_validation_dataset_path": "",
    "max_epochs": 10,
    "maximal_gt_keypoint_pixel_distances": "'8 16 32'",  # quotes are need to avoid splitting in list
    "minimal_keypoint_extraction_pixel_distance": 8,
    "precision": 16,
    "seed": 2024,
    # determined based on hparam sweep
    "heatmap_sigma": 8,
    "learning_rate": 0.0003,
    "batch_size": 8,
    ###
    # "wandb_entity": "tlips",
    "wandb_project": "dsd-paper",
    "wandb_name": None,
}


MUG_DICT = DEFAULT_DICT.copy()
MUG_DICT["keypoint_channel_configuration"] = "bottom:handle:top"
MUG_DICT["json_test_dataset_path"] = str(REAL_MUGS_TEST_DATASET)
MUG_DICT["json_validation_dataset_path"] = str(REAL_MUGS_VAL_DATASET)

SHOE_DICT = DEFAULT_DICT.copy()
SHOE_DICT["keypoint_channel_configuration"] = "nose:heel:top"
SHOE_DICT["json_test_dataset_path"] = str(REAL_SHOES_TEST_DATASET)
SHOE_DICT["json_validation_dataset_path"] = str(REAL_SHOES_VAL_DATASET)


TSHIRT_DICT = DEFAULT_DICT.copy()
TSHIRT_DICT[
    "keypoint_channel_configuration"
] = "shoulder_left:neck_left:neck_right:shoulder_right:sleeve_right_top:sleeve_right_bottom:armpit_right:waist_right:waist_left:armpit_left:sleeve_left_bottom:sleeve_left_top"
TSHIRT_DICT["json_test_dataset_path"] = str(REAL_TSHIRTS_TEST_DATASET)
TSHIRT_DICT["json_validation_dataset_path"] = str(REAL_TSHIRTS_VAL_DATASET)


def _create_command(arg_dict):
    command = COMMAND
    for k, v in arg_dict.items():
        command += f" --{k} {v}"
    return command


def train_on_real():

    epochs = 100

    # mugs
    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-real"
    mug_dict["json_dataset_path"] = str(REAL_MUGS_TRAIN_DATASET)
    mug_dict["ap_epoch_freq"] = 5
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    subprocess.run(command, shell=True)

    # shoes
    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-real"
    shoe_dict["json_dataset_path"] = str(REAL_SHOES_TRAIN_DATASET)
    shoe_dict["ap_epoch_freq"] = 5

    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    # tshirts
    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-real"
    tshirt_dict["json_dataset_path"] = str(REAL_TSHIRTS_TRAIN_DATASET)
    tshirt_dict["ap_epoch_freq"] = 5
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)


def train_on_random_textures():

    epochs = 30

    # mugs
    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-random-textures"
    mug_dict["json_dataset_path"] = str(RANDOM_TEXTURE_BASELINE_MUG_DATASET)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    subprocess.run(command, shell=True)

    # # shoes
    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-random-textures"
    shoe_dict["json_dataset_path"] = str(RANDOM_TEXTURE_BASELINE_SHOE_DATASET)

    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    # tshirts
    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-random-textures"
    tshirt_dict["json_dataset_path"] = str(RANDOM_TEXTURE_BASELINE_TSHIRT_DATASET)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)


def train_on_random_textures_no_table():

    epochs = 30

    # mugs
    mug_dict = MUG_DICT.copy()
    mug_dict["wandb_name"] = "mugs-random-textures-no-table"
    mug_dict["json_dataset_path"] = str(RANDOM_TEXTURE_BASELINE_MUG_NO_TABLE_DATASET)
    mug_dict["max_epochs"] = epochs
    command = _create_command(mug_dict)
    subprocess.run(command, shell=True)

    # # shoes
    shoe_dict = SHOE_DICT.copy()
    shoe_dict["wandb_name"] = "shoes-random-textures-no-table"
    shoe_dict["json_dataset_path"] = str(RANDOM_TEXTURE_BASELINE_SHOE_NO_TABLE_DATASET)

    shoe_dict["max_epochs"] = epochs
    command = _create_command(shoe_dict)
    subprocess.run(command, shell=True)

    # tshirts
    tshirt_dict = TSHIRT_DICT.copy()
    tshirt_dict["wandb_name"] = "tshirts-random-textures-no-table"
    tshirt_dict["json_dataset_path"] = str(RANDOM_TEXTURE_BASELINE_TSHIRT_NO_TABLE_DATASET)
    tshirt_dict["max_epochs"] = epochs
    command = _create_command(tshirt_dict)
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    # train_on_real()
    # train_on_random_textures()
    train_on_random_textures_no_table()
