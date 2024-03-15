class WandbRuns:
    real = {"dsd": "tlips/dsd-mugs-cvpr/runs/1myo185a", "kpam": "tlips/dsd-mugs-cvpr/runs/bsv40dl4"}
    main = {
        "ControlNetFromDepthRenderer_ccs=1.5": "tlips/dsd-mugs-cvpr/runs/037wt4ly",
    }

    model_comparison_1_stage = {
        "ControlNetFromDepthRenderer_ccs=1.5": "tlips/dsd-mugs-cvpr/runs/s4yfql9w",
        "ControlNetTXTFromDepthRenderer_ccs=1.5": "tlips/dsd-mugs-cvpr/runs/l95jg7ji",
        "SD2FromDepthRenderer": "tlips/dsd-mugs-cvpr/runs/2esztwoc",
        "SD2InpaintingRenderer": "tlips/dsd-mugs-cvpr/runs/gcos7d99",
        "SD15RealisticCheckpointControlNetFromDepthRenderer_ccs=1.5": "tlips/dsd-mugs-cvpr/runs/u7wctap2",
        "SD15RealisticCheckpointControlNetTXTFromDepthRenderer_ccs=1.5": "tlips/dsd-mugs-cvpr/runs/3wimzl7w",
        "SDXLControlNetTXTFromDepthRenderer_ccs=1.5": "tlips/dsd-mugs-cvpr/runs/ybpvgfnv",
    }
    ccs_comparison = {
        "ControlNetFromDepthRenderer_ccs=0.5": "tlips/dsd-mugs-cvpr/runs/ejj3xcix",
        "ControlNetFromDepthRenderer_ccs=1.0": "tlips/dsd-mugs-cvpr/runs/0lcip8xx",
        "ControlNetFromDepthRenderer_ccs=1.3": "tlips/dsd-mugs-cvpr/runs/xw12naon",
        "ControlNetFromDepthRenderer_ccs=1.5": "tlips/dsd-mugs-cvpr/runs/s4yfql9w",
        "ControlNetFromDepthRenderer_ccs=2.0": "tlips/dsd-mugs-cvpr/runs/msdtjts3",
        "ControlNetFromDepthRenderer_ccs=2.5": "tlips/dsd-mugs-cvpr/runs/k8hpbsfj",
        "ControlNetFromDepthRenderer_ccs=3.0": "tlips/dsd-mugs-cvpr/runs/s95mb7p8",
    }

    model_comparison_2_stage = {
        "2stage:crop=Cropped:ControlNetFromDepthRenderer_ccs=1.5,_margin=10,_only_change_mask=True,_inp=SD2InpaintingRenderer,_dilation=0": "tlips/dsd-mugs-cvpr/runs/lb0b50ee",
        "2stage:crop=Cropped:ControlNetFromDepthRenderer_ccs=1.5,_margin=10,_only_change_mask=True,_inp=SD2InpaintingRenderer,_dilation=1": "tlips/dsd-mugs-cvpr/runs/tma1ee4l",
        "2stage:crop=Cropped:ControlNetFromDepthRenderer_ccs=1.5,_margin=10,_only_change_mask=True,_inp=SD2InpaintingRenderer,_dilation=2": "tlips/dsd-mugs-cvpr/runs/z7itzs4i",
        "2stage:crop=Cropped:ControlNetFromDepthRenderer_ccs=1.5,_margin=10,_only_change_mask=True,_inp=SD2InpaintingRenderer,_dilation=5": "tlips/dsd-mugs-cvpr/runs/hfbkdhac",
        "2stage:crop=Cropped:ControlNetFromDepthRenderer_ccs=1.5,_margin=10,_only_change_mask=True,_inp=SD15InpaintingRenderer,_dilation=1": "tlips/dsd-mugs-cvpr/runs/iq52x8tm",
        "2stage:crop=Cropped:ControlNetFromDepthRenderer_ccs=1.5,_margin=15,_only_change_mask=False,_inp=SD2InpaintingRenderer,_dilation=3": "tlips/dsd-mugs-cvpr/runs/8igmlx0s",
    }

    all_checkpoint_dict = {
        "real": real,
        "large-run": main,
        "model-comparison-1-stage": model_comparison_1_stage,
        "ccs-comparison": ccs_comparison,
        "stage-2-comparison": model_comparison_2_stage,
    }


import pandas as pd
import wandb


def fetch_full_data_from_wandb(run_path, key1, key2):
    api = wandb.Api()
    run = api.run(run_path)
    history = run.scan_history(keys=[key1, key2], page_size=10000)

    key1_list = []
    key2_list = []

    for row in history:
        key1_list.append(row[key1])
        key2_list.append(row[key2])

    dictionary = {key1: key1_list, key2: key2_list}
    df = pd.DataFrame.from_dict(dictionary)
    return df


def get_AP_metrics(run):
    df = fetch_full_data_from_wandb(run, "epoch", "test/meanAP/meanAP")
    mAPmean = float(df["test/meanAP/meanAP"][0])
    return mAPmean


if __name__ == "__main__":
    print(get_AP_metrics(WandbRuns.real["dsd"]))
