import wandb

REAL_MUGS_RUN = "tlips/dsd-paper-yolo/runs/7wgj299e"
REAL_SHOES_RUN = "tlips/dsd-paper-yolo/runs/ksel1bg7"
REAL_TSHIRTS_RUN = "tlips/dsd-paper-yolo/runs/8qgu6l4s"

# random textures
RANDOM_TEXTURES_MUGS_RUN = "tlips/dsd-paper-yolo/runs/ola7i6sz"
RANDOM_TEXTURES_SHOES_RUN = "tlips/dsd-paper-yolo/runs/mjd5800n"
RANDOM_TEXTURES_TSHIRTS_RUN = "tlips/dsd-paper-yolo/runs/vy3g3o5g"

## PROMPT experiment

PROMPT_CLASSNAME_MUGS_RUN = "tlips/dsd-paper-yolo/runs/nvndrirc"
PROMPT_CLASSNAME_SHOES_RUN = "tlips/dsd-paper-yolo/runs/cfiune2t"
PROMPT_CLASSNAME_TSHIRTS_RUN = "tlips/dsd-paper-yolo/runs/wpz1k362"

PROMPT_GEMINI_MUGS_RUN = "tlips/dsd-paper-yolo/runs/nl4pqdu1"
PROMPT_GEMINI_SHOES_RUN = "tlips/dsd-paper-yolo/runs/75dm6fa5"
PROMPT_GEMINI_TSHIRTS_RUN = "tlips/dsd-paper-yolo/runs/5v57ni2p"

PROMPT_BLIP_MUGS_RUN = "tlips/dsd-paper-yolo/runs/4ns88tfn"
PROMPT_BLIP_SHOES_RUN = "tlips/dsd-paper-yolo/runs/e9yv0bez"
PROMPT_BLIP_TSHIRTS_RUN = "tlips/dsd-paper-yolo/runs/6xsh00r9"

TWO_STAGE_BASELINE_MUGS_RUN = "tlips/dsd-paper-yolo/runs/axed16b5"
TWO_STAGE_BASELINE_SHOES_RUN = "tlips/dsd-paper-yolo/runs/m940xmmm"
TWO_STAGE_BASELINE_TSHIRTS_RUN = "tlips/dsd-paper-yolo/runs/2oq55vmh"


def get_model_checkpoint_artifact_from_run(run_id):
    api = wandb.Api()
    run = api.run(run_id)
    model_artifact = [a for a in run.logged_artifacts() if a.type == "model"][0]
    return model_artifact


def get_bbox_mAP_from_run(run_id):
    api = wandb.Api()
    run = api.run(run_id)
    metrics = run.summary_metrics
    return metrics["test/bbox_mAP"]


def get_seg_mAP_from_run(run_id):
    api = wandb.Api()
    run = api.run(run_id)
    metrics = run.summary_metrics
    return metrics["test/seg_mAP"]


def generate_meanAP_dict(all_runs, get=get_seg_mAP_from_run):

    meanAP_dict = {}
    for name, run in all_runs.items():
        meanAP = get(run)
        meanAP_dict[name] = meanAP
    return meanAP_dict


if __name__ == "__main__":

    from pathlib import Path

    file_path = Path(__file__)
    results_dir = file_path.parent / "run_metrics"
    import json

    # get all global variables with 'run' in their name
    all_runs = {k: v for k, v in globals().items() if "RUN" in k}

    print("verifying run ids are consistent with names")
    for k, v in all_runs.items():
        if v:
            run = wandb.Api().run(v)
            print(f"{k}: {run.name}")

    # get the mAPs
    meanAP_dict = generate_meanAP_dict(all_runs, get=get_seg_mAP_from_run)
    with open(str(results_dir / "seg_meanAP_dict.json"), "w") as f:
        json.dump(meanAP_dict, f)

    meanAP_dict = generate_meanAP_dict(all_runs, get=get_bbox_mAP_from_run)
    with open(str(results_dir / "bbox_meanAP_dict.json"), "w") as f:
        json.dump(meanAP_dict, f)
