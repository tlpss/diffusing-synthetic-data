class Checkpoints:
    real = {"dsd": "tlips/dsd-mugs-cvpr/model-1myo185a:v0", "kpam": "tlips/dsd-mugs-cvpr/model-bsv40dl4:v0"}
    main = {
        "ControlNetFromDepthRenderer_ccs=1.5": "tlips/dsd-mugs-cvpr/model-037wt4ly:v0",
    }

    model_comparison_1_stage = {
        "ControlNetFromDepthRenderer_ccs=1.5": "tlips/dsd-mugs-cvpr/model-s4yfql9w:v0",
        "ControlNetTXTFromDepthRenderer_ccs=1.5": "tlips/dsd-mugs-cvpr/model-l95jg7ji:v0",
        "SD2FromDepthRenderer": "tlips/dsd-mugs-cvpr/model-2esztwoc:v0",
        "SD2InpaintingRenderer": "tlips/dsd-mugs-cvpr/model-gcos7d99:v0",
        "SD15RealisticCheckpointControlNetFromDepthRenderer_ccs=1.5": "tlips/dsd-mugs-cvpr/model-u7wctap2:v0",
        "SD15RealisticCheckpointControlNetTXTFromDepthRenderer_ccs=1.5": "tlips/dsd-mugs-cvpr/model-3wimzl7w:v0",
        "SDXLControlNetTXTFromDepthRenderer_ccs=1.5": "tlips/dsd-mugs-cvpr/model-ybpvgfnv:v0",
    }
    ccs_comparison = {
        "ControlNetFromDepthRenderer_ccs=0.5": "tlips/dsd-mugs-cvpr/model-ejj3xcix:v0",
        "ControlNetFromDepthRenderer_ccs=1.0": "tlips/dsd-mugs-cvpr/model-0lcip8xx:v0",
        "ControlNetFromDepthRenderer_ccs=1.3": "tlips/dsd-mugs-cvpr/model-xw12naon:v0",
        "ControlNetFromDepthRenderer_ccs=1.5": "tlips/dsd-mugs-cvpr/model-s4yfql9w:v0",
        "ControlNetFromDepthRenderer_ccs=2.0": "tlips/dsd-mugs-cvpr/model-msdtjts3:v0",
        "ControlNetFromDepthRenderer_ccs=2.5": "tlips/dsd-mugs-cvpr/model-k8hpbsfj:v0",
        "ControlNetFromDepthRenderer_ccs=3.0": "tlips/dsd-mugs-cvpr/model-s95mb7p8:v0",
    }

    model_comparison_2_stage = {
        "2stage:crop=Cropped:ControlNetFromDepthRenderer_ccs=1.5,_margin=10,_only_change_mask=True,_inp=SD2InpaintingRenderer,_dilation=0": "tlips/dsd-mugs-cvpr/model-lb0b50ee:v0",
        "2stage:crop=Cropped:ControlNetFromDepthRenderer_ccs=1.5,_margin=10,_only_change_mask=True,_inp=SD2InpaintingRenderer,_dilation=1": "tlips/dsd-mugs-cvpr/model-tma1ee4l:v0",
        "2stage:crop=Cropped:ControlNetFromDepthRenderer_ccs=1.5,_margin=10,_only_change_mask=True,_inp=SD2InpaintingRenderer,_dilation=2": "tlips/dsd-mugs-cvpr/model-z7itzs4i:v0",
        "2stage:crop=Cropped:ControlNetFromDepthRenderer_ccs=1.5,_margin=10,_only_change_mask=True,_inp=SD2InpaintingRenderer,_dilation=5": "tlips/dsd-mugs-cvpr/model-hfbkdhac:v0",
        "2stage:crop=Cropped:ControlNetFromDepthRenderer_ccs=1.5,_margin=10,_only_change_mask=True,_inp=SD15InpaintingRenderer,_dilation=1": "tlips/dsd-mugs-cvpr/model-iq52x8tm:v0",
        "2stage:crop=Cropped:ControlNetFromDepthRenderer_ccs=1.5,_margin=15,_only_change_mask=False,_inp=SD2InpaintingRenderer,_dilation=3": "tlips/dsd-mugs-cvpr/model-8igmlx0s:v0",
    }

    all_checkpoint_dict = {
        "real": real,
        "large-run": main,
        "model-comparison-1-stage": model_comparison_1_stage,
        "ccs-comparison": ccs_comparison,
        "stage-2-comparison": model_comparison_2_stage,
    }


if __name__ == "__main__":
    print(Checkpoints.Real.dsd)
