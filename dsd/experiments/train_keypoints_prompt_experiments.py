import subprocess

from experiments.train_keypoints import MUG_DICT, SHOE_DICT, TSHIRT_DICT, _create_command
from paths import (  # noqa
    PROMPTS_BLIP_MUG_DATASET,
    PROMPTS_BLIP_SHOE_DATASET,
    PROMPTS_BLIP_TSHIRT_DATASET,
    PROMPTS_CLASSNAME_MUG_DATASET,
    PROMPTS_CLASSNAME_SHOE_DATASET,
    PROMPTS_CLASSNAME_TSHIRT_DATASET,
    PROMPTS_GEMINI_MUG_DATASET,
    PROMPTS_GEMINI_SHOE_DATASET,
    PROMPTS_GEMINI_TSHIRT_DATASET,
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


if __name__ == "__main__":
    train_on_prompts_blip()
    train_on_prompts_gemini()
    train_on_prompts_classname()
