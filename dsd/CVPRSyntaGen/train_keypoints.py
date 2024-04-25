import subprocess

COMMAND = "keypoint-detection train  --augment_train"
from dsd import DATA_DIR

DEFAULT_DICT = {
    "keypoint_channel_configuration": None,
    "accelerator": "gpu",
    "ap_epoch_freq": 1,
    "backbone_type": "MaxVitUnet",
    "devices": 1,
    "early_stopping_relative_threshold": -1,
    "json_dataset_path": "",
    "json_test_dataset_path": "",
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
    "wandb_project": "dsd-mugs-cvpr",
    "wandb_name": None,
}


MUG_DICT = DEFAULT_DICT.copy()
MUG_DICT["keypoint_channel_configuration"] = "bottom:handle:top"
MUG_DICT["json_test_dataset_path"] = str(
    DATA_DIR / "real" / "mugs" / "lab-mugs_resized_512x512" / "lab-mugs_train.json"
)
MUG_DICT["json_validation_dataset_path"] = str(
    DATA_DIR / "real" / "mugs" / "lab-mugs_resized_512x512" / "lab-mugs_val.json"
)


def _create_command(arg_dict):
    command = COMMAND
    for k, v in arg_dict.items():
        command += f" --{k} {v}"
    return command


def train_on_synthetic_dataset(synthetic_dataset_path, default_arg_dict):
    arg_dict = default_arg_dict.copy()
    arg_dict["json_dataset_path"] = '"' + str(synthetic_dataset_path / "annotations.json") + '"'
    arg_dict["wandb_name"] = f"mugs-{synthetic_dataset_path.relative_to(synthetic_dataset_path.parents[1])}"
    command = _create_command(arg_dict)
    print(command)
    subprocess.run(command, shell=True)


def train_on_experiment(experiment_coco_root, default_arg_dict=MUG_DICT):
    default_arg_dict = default_arg_dict.copy()
    for renderer_path in experiment_coco_root.iterdir():
        train_on_synthetic_dataset(renderer_path, default_arg_dict)


def train_on_kpam_dsd():
    arg_dict = MUG_DICT.copy()
    arg_dict["json_dataset_path"] = (
        DATA_DIR / "real" / "mugs" / "kpam-mugs-dsd-train_resized_512x512" / "annotations.json"
    )
    arg_dict["wandb_name"] = "kpam-dsd-mugs"

    command = _create_command(arg_dict)
    subprocess.run(command, shell=True)


def train_on_robot_dsd():
    arg_dict = MUG_DICT.copy()
    arg_dict["json_dataset_path"] = DATA_DIR / "real" / "mugs" / "dsd-mugs-robot" / "annotations.json"
    arg_dict["wandb_name"] = "robot-dsd-mugs"
    arg_dict["max_epochs"] = 20
    command = _create_command(arg_dict)
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    # train_on_robot_dsd()
    # train_on_kpam_dsd()
    # train_on_experiment(DATA_DIR / "diffusion_renders" / "coco"/ "mugs" / "cvpr" / "model-comparison-1-stage")

    # delete all spaces from the path to avoid issues with the shell

    # for p in (DATA_DIR / "diffusion_renders" / "coco"/ "mugs" / "cvpr" / "stage-2-comparison").iterdir():
    #     p.rename(p.parent / p.name.replace(" ", "_"))
    # train_on_experiment(DATA_DIR / "diffusion_renders" / "coco"/ "mugs" / "cvpr" / "stage-2-comparison")
    # train_on_synthetic_dataset(DATA_DIR / "diffusion_renders" / "coco"/ "mugs" / "cvpr" / "stage-2-comparison" / "2stage:crop=Cropped:ControlNetFromDepthRenderer_ccs=1.5,_margin=10,_only_change_mask=True,_inp=SD2InpaintingRenderer,_dilation=1",MUG_DICT)
    train_on_experiment(DATA_DIR / "diffusion_renders" / "coco" / "mugs" / "cvpr" / "large-run-2-stage")
