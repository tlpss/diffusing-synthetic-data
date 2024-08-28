import wandb

# REAL

REAL_MUGS_RUN = "tlips/dsd-paper/runs/uwakr467"
REAL_SHOES_RUN = "tlips/dsd-paper/runs/l87ah1f9"
REAL_TSHIRTS_RUN = "tlips/dsd-paper/runs/jvjqj9o8"

# random textures

RANDOM_TEXTURES_MUGS_RUN = "tlips/dsd-paper/runs/kt04wrsb"
RANDOM_TEXTURES_SHOES_RUN = "tlips/dsd-paper/runs/c26497pa"
RANDOM_TEXTURES_TSHIRTS_RUN = "tlips/dsd-paper/runs/pff2qijf"

## PROMPT experiment

PROMPT_CLASSNAME_MUGS_RUN = "tlips/dsd-paper/runs/kpchbiwr"
PROMPT_CLASSNAME_SHOES_RUN = "tlips/dsd-paper/runs/7z4g4skw"
PROMPT_CLASSNAME_TSHIRTS_RUN = "tlips/dsd-paper/runs/dgcloc6a"

PROMPT_GEMINI_MUGS_RUN = "tlips/dsd-paper/runs/s5gwjor3"
PROMPT_GEMINI_SHOES_RUN = "tlips/dsd-paper/runs/dl6ksv7b"
PROMPT_GEMINI_TSHIRTS_RUN = "tlips/dsd-paper/runs/29zt96n2"

PROMPT_BLIP_MUGS_RUN = "tlips/dsd-paper/runs/x3bmzjck"
PROMPT_BLIP_SHOES_RUN = "tlips/dsd-paper/runs/6c8a4c7j"
PROMPT_BLIP_TSHIRTS_RUN = "tlips/dsd-paper/runs/xorggoo7"


## two stage baseline

TWO_STAGE_BASELINE_MUGS_RUN = "tlips/dsd-paper/runs/681l1mbo"
TWO_STAGE_BASELINE_SHOES_RUN = "tlips/dsd-paper/runs/jpx0m49x"
TWO_STAGE_BASELINE_TSHIRTS_RUN = "tlips/dsd-paper/runs/4pxny9v7"


## No table diffusion
NO_TABLE_ONE_STAGE_MUGS_RUN = "tlips/dsd-paper/runs/f7axdezq"
NO_TABLE_ONE_STAGE_SHOES_RUN = "tlips/dsd-paper/runs/zq4gxiqq"
NO_TABLE_ONE_STAGE_TSHIRTS_RUN = "tlips/dsd-paper/runs/hhcmazbm"

## no table random textures
RANDOM_TEXTURES_NO_TABLE_MUGS_RUN = "tlips/dsd-paper/runs/2hdpahys"
RANDOM_TEXTURES_NO_TABLE_SHOES_RUN = "tlips/dsd-paper/runs/mtifa9qa"
RANDOM_TEXTURES_NO_TABLE_TSHIRTS_RUN = "tlips/dsd-paper/runs/gxgzakg3"

## three stage
THREE_STAGE_MUGS_RUN = "tlips/dsd-paper/runs/tuq0m8dw"
THREE_STAGE_SHOES_RUN = "tlips/dsd-paper/runs/nlm4a93p"
THREE_STAGE_TSHIRTS_RUN = "tlips/dsd-paper/runs/jzxdh9cz"

## Dual inpaint
DUAL_INPAINTING_MUGS_RUN = "tlips/dsd-paper/runs/ssqkbrpn"
DUAL_INPAINTING_TSHIRTS_RUN = "tlips/dsd-paper/runs/e57cbyn5"
DUAL_INPAINTING_SHOES_RUN = "tlips/dsd-paper/runs/zkclun8j"
## img 2 img

IMG2IMG_MUGS_RUN = "tlips/dsd-paper/runs/0c9kpjey"
IMG2IMG_SHOES_RUN = "tlips/dsd-paper/runs/14zptl31"
IMG2IMG_TSHIRTS_RUN = "tlips/dsd-paper/runs/4pv4bx0v"


### CCS

# 0.5
CCS_O5_MUGS_RUN = "tlips/dsd-paper/runs/640qrpo8"
CCS_O5_SHOES_RUN = "tlips/dsd-paper/runs/g5hx4884"
CCS_O5_TSHIRTS_RUN = "tlips/dsd-paper/runs/muqz1ybl"

# 1.0
CCS_10_MUGS_RUN = "tlips/dsd-paper/runs/zn71hy8d"
CCS_10_SHOES_RUN = "tlips/dsd-paper/runs/q6wbhpa5"
CCS_10_TSHIRTS_RUN = "tlips/dsd-paper/runs/dddtjfq8"

# 1.5
CCS_15_MUGS_RUN = "tlips/dsd-paper/runs/8bvtpqw9"
CCS_15_SHOES_RUN = "tlips/dsd-paper/runs/rcei1ugz"
CCS_15_TSHIRTS_RUN = "tlips/dsd-paper/runs/ivfco0pd"

# 2.0

CCS_20_MUGS_RUN = "tlips/dsd-paper/runs/fx7vd0eg"
CCS_20_SHOES_RUN = "tlips/dsd-paper/runs/77owwrdr"
CCS_20_TSHIRTS_RUN = "tlips/dsd-paper/runs/01zkyyu6"

# 2.5

CCS_25_MUGS_RUN = "tlips/dsd-paper/runs/npfclpbf"
CCS_25_SHOES_RUN = "tlips/dsd-paper/runs/8feoocst"
CCS_25_TSHIRTS_RUN = "tlips/dsd-paper/runs/pvuxkvfi"


def get_model_checkpoint_artifact_from_run(run_id):
    api = wandb.Api()
    run = api.run(run_id)
    model_artifact = [a for a in run.logged_artifacts() if a.type == "model"][0]
    return model_artifact


def get_meanAP_from_run(run_id):
    api = wandb.Api()
    run = api.run(run_id)
    metrics = run.summary_metrics
    return metrics["test/meanAP"]


if __name__ == "__main__":

    # get all global variables with 'run' in their name
    all_runs = {k: v for k, v in globals().items() if "RUN" in k}

    print("verifying run ids are consistent with names")
    for k, v in all_runs.items():
        if v:
            run = wandb.Api().run(v)
            print(f"{k}: {run.name}")
