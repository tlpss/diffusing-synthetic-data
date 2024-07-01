from experiments.keypoint_wandb_runs import get_meanAP_from_run, get_model_checkpoint_artifact_from_run
from experiments.paths import REAL_MUGS_TEST_DATASET, REAL_SHOES_TEST_DATASET, REAL_TSHIRTS_TEST_DATASET
from keypoint_detection.utils.load_checkpoints import get_model_from_wandb_checkpoint

from dsd.generate_coco_datasets_from_diffusion_renders import mug_category, shoe_category, tshirt_category
from dsd.image_metrics.keypoints_average_error import calculate_average_error_for_dataset


def generate_meanAP_dict(all_runs):

    meanAP_dict = {}
    for name, run in all_runs.items():
        meanAP = get_meanAP_from_run(run)
        meanAP_dict[name] = meanAP
    return meanAP_dict


def generate_AKD_dict(all_runs):

    AKD_dict = {}
    median_KD_dict = {}
    for k, v in all_runs.items():
        if v:
            model_artifact = get_model_checkpoint_artifact_from_run(v)
            model = get_model_from_wandb_checkpoint(model_artifact)

            if "mug" in k.lower():
                dataset_json_path = REAL_MUGS_TEST_DATASET
                channel_config = mug_category.keypoints
            elif "shoe" in k.lower():
                dataset_json_path = REAL_SHOES_TEST_DATASET
                channel_config = shoe_category.keypoints
            elif "tshirt" in k.lower():
                dataset_json_path = REAL_TSHIRTS_TEST_DATASET
                channel_config = tshirt_category.keypoints
            else:
                raise ValueError("Dataset not found")
            channel_config = [[c] for c in channel_config]
            akd, median = calculate_average_error_for_dataset(
                model,
                dataset_json_path,
                channel_config,
                False,
            )
            AKD_dict[k] = akd
            median_KD_dict[k] = median
    return AKD_dict, median_KD_dict


if __name__ == "__main__":
    from pathlib import Path

    from experiments.keypoint_wandb_runs import *  # noqa

    run_metric_dir = Path(__file__).parent / "run_metrics"

    all_runs = {k: v for k, v in globals().items() if "RUN" in k}

    # get first two entries of dict
    # all_runs = {k: v for k, v in list(all_runs.items())[:2]}

    meanAP_dict = generate_meanAP_dict(all_runs)
    AKD_dict, median_KD_dict = generate_AKD_dict(all_runs)

    # save to file
    import json

    with open(str(run_metric_dir / "meanAP_dict.json"), "w") as f:
        json.dump(meanAP_dict, f)

    with open(str(run_metric_dir / "AKD_dict.json"), "w") as f:
        json.dump(AKD_dict, f)

    with open(str(run_metric_dir / "median_KD_dict.json"), "w") as f:
        json.dump(median_KD_dict, f)
