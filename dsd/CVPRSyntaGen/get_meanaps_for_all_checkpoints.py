import copy

from CVPRSyntaGen.experiments.wandb_runs import WandbRuns, get_AP_metrics

if __name__ == "__main__":

    # create a nested dict for all checkpoints:
    all_checkpoints = WandbRuns.all_checkpoint_dict
    mae_dict = copy.deepcopy(all_checkpoints)

    for experiment_name, ckpt_dict in all_checkpoints.items():
        for run_name, checkpoint in ckpt_dict.items():
            mAP = get_AP_metrics(checkpoint)
            mae_dict[experiment_name][run_name] = get_AP_metrics(checkpoint)

    # write to json file
    import json
    import pathlib

    cvpr_dir = pathlib.Path(__file__).parent

    with open(str(cvpr_dir / "mAP.json"), "w") as f:
        json.dump(mae_dict, f)
