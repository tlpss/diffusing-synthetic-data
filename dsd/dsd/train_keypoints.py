import subprocess

COMMAND = "keypoint-detection train  --augment_train"
from dsd import DATA_DIR

DEFAULT_DICT = {
    "keypoint_channel_configuration": None,
    "accelerator": "gpu",
    "ap_epoch_freq": 5,
    # "check_val_every_n_epoch": 10,
    "backbone_type": "MaxVitUnet",
    "devices": 1,
    "early_stopping_relative_threshold": -1,
    "json_dataset_path": "",
    "json_test_dataset_path": "",
    # "json_validation_dataset_path": "",
    "max_epochs": 25,
    "maximal_gt_keypoint_pixel_distances": "'8 16 32'",  # quotes are need to avoid splitting in list
    "minimal_keypoint_extraction_pixel_distance": 2,
    "precision": 16,
    "seed": 2024,
    # determined based on hparam sweep
    "heatmap_sigma": 3,
    "learning_rate": 0.0003,
    "batch_size": 8,
    ###
    # "wandb_entity": "tlips",
    "wandb_project": "dsd-mugs",
    "wandb_name": None,
}


MUG_DICT = DEFAULT_DICT.copy()
MUG_DICT["keypoint_channel_configuration"] = "bottom:handle:top"
MUG_DICT["json_test_dataset_path"] = str(DATA_DIR / "real" / "mugs" / "lab-mugs_resized_512x512" / "lab-mugs_val.json")


def _create_command(arg_dict):
    command = COMMAND
    for k, v in arg_dict.items():
        command += f" --{k} {v}"
    return command


def train_on_synthetic_dataset(synthetic_dataset_path, default_arg_dict):
    arg_dict = default_arg_dict.copy()
    arg_dict["json_dataset_path"] = str(synthetic_dataset_path / "annotations.json")
    # arg_dict["json_validation_dataset_path"] = str(DATA_DIR / "real"/"mugs"/"lab-mugs_resized_512x512"/"lab-mugs_val.json")
    arg_dict["wandb_name"] = f"mugs-{synthetic_dataset_path.relative_to(synthetic_dataset_path.parents[1])}"
    command = _create_command(arg_dict)
    subprocess.run(command, shell=True)


def train_on_experiment(experiment_coco_root, default_arg_dict=MUG_DICT):
    default_arg_dict = default_arg_dict.copy()
    for renderer_path in experiment_coco_root.iterdir():
        train_on_synthetic_dataset(renderer_path, default_arg_dict)


def train_on_real(real_train_path, real_val_path, default_arg_dict=MUG_DICT):
    arg_dict = default_arg_dict.copy()
    arg_dict["json_dataset_path"] = str(real_train_path)
    arg_dict["json_test_dataset_path"] = str(real_val_path)
    arg_dict["wandb_name"] = "mugs-real"
    arg_dict["validation_split_ratio"] = 0.15
    arg_dict["max_epochs"] = 100
    command = _create_command(arg_dict)
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    train_on_real(
        DATA_DIR / "real" / "mugs" / "lab-mugs_resized_512x512" / "lab-mugs_train.json",
        DATA_DIR / "real" / "mugs" / "lab-mugs_resized_512x512" / "lab-mugs_val.json",
    )
    # train_on_experiment(DATA_DIR / "diffusion_renders" / "coco"  / "mugs" / "run_3")
    # train_on_synthetic_dataset(DATA_DIR / "diffusion_renders" / "coco"  / "mugs" / "run_3" / "ControlNetFromDepthRenderer_ccs=1.5", MUG_DICT.copy())
