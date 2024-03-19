import copy

from CVPRSyntaGen.experiments.checkpoints import Checkpoints
from CVPRSyntaGen.keypoints_average_error import calculate_average_error_for_dataset
from keypoint_detection.utils.load_checkpoints import get_model_from_wandb_checkpoint

from dsd import DATA_DIR

if __name__ == "__main__":
    dataset_json_path = DATA_DIR / "real" / "mugs" / "lab-mugs_resized_512x512" / "lab-mugs_train.json"
    visible_only = False

    # create a nested dict for all checkpoints:
    all_checkpoints = Checkpoints.all_checkpoint_dict
    mae_dict = copy.deepcopy(all_checkpoints)

    for experiment_name, ckpt_dict in all_checkpoints.items():
        for checkpoint_name, checkpoint in ckpt_dict.items():
            print(f"Calculating Average L2 error for {experiment_name} - {checkpoint_name}")
            model = get_model_from_wandb_checkpoint(checkpoint).cuda()
            avg_errors = calculate_average_error_for_dataset(
                model, dataset_json_path, [["bottom"], ["handle"], ["top"]], visible_only
            )
            mae_dict[experiment_name][checkpoint_name] = avg_errors

    # write to json file
    import json
    import pathlib

    cvpr_dir = pathlib.Path(__file__).parent

    with open(str(cvpr_dir / f"avg-error{'_visible_only' if visible_only else ''}.json"), "w") as f:
        json.dump(mae_dict, f)
