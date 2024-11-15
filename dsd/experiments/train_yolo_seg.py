from datetime import datetime
from pathlib import Path

import wandb
from ultralytics import YOLO, settings

from dsd import DATA_DIR

settings.update({"datasets_dir": str(DATA_DIR / "yolo")})
# rundir
settings.update({"runs_dir": str(DATA_DIR.parent / "yolo-logs")})
settings.update({"weights_dir": str(DATA_DIR.parent / "yolo-logs")})


# create temp yolo data.yaml file


def create_yolo_seg_data_yaml(train_dataset_path, val_dataset_path, class_name, filename):

    train_dataset_path = Path(train_dataset_path)
    val_dataset_path = Path(val_dataset_path)
    for path in [train_dataset_path, val_dataset_path]:
        if path is not None:
            # if absolute path, convert to relative path
            if path.is_absolute():
                path = path.relative_to(DATA_DIR / "yolo")

    data = f"""
    path: .
    train: {str(train_dataset_path)}
    val: {str(val_dataset_path)}

    names:
        0: {class_name}
    """
    # ^ hack to fix that tshirt dataset has multiple class labels, and tshirts have id=2

    with open(filename, "w") as f:
        f.write(data)
    print()


def train_and_test_yolo_seg(train_name, train_dataset, val_dataset, test_dataset, category):

    wandb.init(project="dsd-paper-yolo", name=train_name)

    # disable wandb finish to keep ultlralytics from finishing the run
    WANDB_FINISH = wandb.run.finish
    wandb.run.finish = lambda: None

    train_dataset = str(train_dataset)
    val_dataset = str(val_dataset)
    test_dataset = str(test_dataset)

    # append wandb run id to train_name to make it unique and avoid suffix by ultralytics
    yolo_train_name = f"{train_name}_{wandb.run.id}"

    # create a seg model pretrained on COCO
    model = YOLO("yolov8s-seg")

    FILENAME = f"{datetime.now()}_data.yaml"
    create_yolo_seg_data_yaml(train_dataset, val_dataset, category, FILENAME)

    model.train(data=FILENAME, epochs=100, imgsz=512, name=yolo_train_name, batch=32, workers=2)

    # evaluate the model
    # load best checkpoint
    model = YOLO(f"{DATA_DIR.parent}/yolo-logs/segment/{yolo_train_name}/weights/best.pt")

    # set test dataset as val to evaluate
    create_yolo_seg_data_yaml(train_dataset, test_dataset, category, FILENAME)
    test_results = model.val(data=FILENAME)
    all_aps = test_results.seg.all_ap
    m_ap = test_results.seg.map

    if wandb.run:
        wandb.log({"test/seg_mAP": m_ap})
        wandb.log({"test/bbox_mAP": test_results.box.map})

    print(f"mAP: {m_ap}")
    print("all APs")

    thresholds = ["0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95"]
    for i, ap in enumerate(all_aps[0]):
        wandb.log({f"test/seg/AP_{thresholds[i]}": ap})

    # remove the temp yolo data.yaml file
    import os

    os.remove(FILENAME)

    WANDB_FINISH()


if __name__ == "__main__":  # noqa C901
    from generate_yolo_datasets import coco_path_to_yolo_path, scale_experiment_coco_to_yolo
    from paths import (  # noqa
        DUAL_INPAINT_MUGS_DATASET,
        DUAL_INPAINT_SHOES_DATASET,
        DUAL_INPAINT_TSHIRTS_DATASET,
        IMG2IMG_MUGS_DATASET,
        IMG2IMG_SHOES_DATASET,
        IMG2IMG_TSHIRTS_DATASET,
        ONE_STAGE_LARGE_MUG_DATASET,
        ONE_STAGE_LARGE_SHOE_DATASET,
        ONE_STAGE_LARGE_TSHIRT_DATASET,
        ONE_STAGE_NO_TABLE_MUG_DATASET,
        ONE_STAGE_NO_TABLE_SHOE_DATASET,
        ONE_STAGE_NO_TABLE_TSHIRT_DATASET,
        PROMPTS_BLIP_MUG_DATASET,
        PROMPTS_BLIP_SHOE_DATASET,
        PROMPTS_BLIP_TSHIRT_DATASET,
        PROMPTS_CLASSNAME_MUG_DATASET,
        PROMPTS_CLASSNAME_SHOE_DATASET,
        PROMPTS_CLASSNAME_TSHIRT_DATASET,
        PROMPTS_GEMINI_MUG_DATASET,
        PROMPTS_GEMINI_SHOE_DATASET,
        PROMPTS_GEMINI_TSHIRT_DATASET,
        RANDOM_TEXTURE_BASELINE_MUG_DATASET,
        RANDOM_TEXTURE_BASELINE_MUG_NO_TABLE_DATASET,
        RANDOM_TEXTURE_BASELINE_SHOE_DATASET,
        RANDOM_TEXTURE_BASELINE_SHOE_NO_TABLE_DATASET,
        RANDOM_TEXTURE_BASELINE_TSHIRT_DATASET,
        RANDOM_TEXTURE_BASELINE_TSHIRT_NO_TABLE_DATASET,
        RANDOM_TEXTURE_MUG_LARGE_DATASET,
        RANDOM_TEXTURE_SHOE_LARGE_DATASET,
        RANDOM_TEXTURE_TSHIRT_LARGE_DATASET,
        REAL_MUGS_TEST_DATASET,
        REAL_MUGS_TRAIN_DATASET,
        REAL_MUGS_VAL_DATASET,
        REAL_SHOES_TEST_DATASET,
        REAL_SHOES_TRAIN_DATASET,
        REAL_SHOES_VAL_DATASET,
        REAL_TSHIRTS_TEST_DATASET,
        REAL_TSHIRTS_TRAIN_DATASET,
        REAL_TSHIRTS_VAL_DATASET,
        THREE_STAGE_MUG_DATASET,
        THREE_STAGE_SHOE_DATASET,
        THREE_STAGE_TSHIRT_DATASET,
        TWO_STAGE_BASELINE_MUG_DATASET,
        TWO_STAGE_BASELINE_SHOE_DATASET,
        TWO_STAGE_BASELINE_TSHIRT_DATASET,
    )

    def real_dataset_to_masked(x):
        return Path(str(x).replace(".json", "_sam_masked.json"))

    def train_on_real_datasets():
        # tshirts, shoes, mugs
        train_and_test_yolo_seg(
            "tshirts-real",
            coco_path_to_yolo_path(REAL_TSHIRTS_TRAIN_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
            "tshirt",
        )

        train_and_test_yolo_seg(
            "shoes-real",
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TRAIN_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
            "shoe",
        )

        train_and_test_yolo_seg(
            "mugs-real",
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TRAIN_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
            "mug",
        )

    def train_on_random_texture_baselines():
        # tshirts, shoes, mugs
        train_and_test_yolo_seg(
            "tshirts-random",
            coco_path_to_yolo_path(RANDOM_TEXTURE_BASELINE_TSHIRT_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
            "tshirt",
        )

        train_and_test_yolo_seg(
            "shoes-random",
            coco_path_to_yolo_path(RANDOM_TEXTURE_BASELINE_SHOE_DATASET),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
            "shoe",
        )

        train_and_test_yolo_seg(
            "mugs-random",
            coco_path_to_yolo_path(RANDOM_TEXTURE_BASELINE_MUG_DATASET),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
            "mug",
        )

    def train_on_prompt_experiments():
        # tshirts, shoes, mugs

        train_and_test_yolo_seg(
            "tshirts-prompts-blip",
            coco_path_to_yolo_path(PROMPTS_BLIP_TSHIRT_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
            "tshirt",
        )
        train_and_test_yolo_seg(
            "tshirts-prompts-gemini",
            coco_path_to_yolo_path(PROMPTS_GEMINI_TSHIRT_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
            "tshirt",
        )
        train_and_test_yolo_seg(
            "tshirts-prompts-classname",
            coco_path_to_yolo_path(PROMPTS_CLASSNAME_TSHIRT_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
            "tshirt",
        )

        train_and_test_yolo_seg(
            "shoes-prompts-blip",
            coco_path_to_yolo_path(PROMPTS_BLIP_SHOE_DATASET),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
            "shoe",
        )
        train_and_test_yolo_seg(
            "shoes-prompts-gemini",
            coco_path_to_yolo_path(PROMPTS_GEMINI_SHOE_DATASET),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
            "shoe",
        )
        train_and_test_yolo_seg(
            "shoes-prompts-classname",
            coco_path_to_yolo_path(PROMPTS_CLASSNAME_SHOE_DATASET),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
            "shoe",
        )

        train_and_test_yolo_seg(
            "mugs-prompts-blip",
            coco_path_to_yolo_path(PROMPTS_BLIP_MUG_DATASET),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
            "mug",
        )
        train_and_test_yolo_seg(
            "mugs-prompts-gemini",
            coco_path_to_yolo_path(PROMPTS_GEMINI_MUG_DATASET),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
            "mug",
        )
        train_and_test_yolo_seg(
            "mugs-prompts-classname",
            coco_path_to_yolo_path(PROMPTS_CLASSNAME_MUG_DATASET),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
            "mug",
        )

    def train_on_two_stage_baseline():
        # tshirts, shoes, mugs
        train_and_test_yolo_seg(
            "shoes-two-stage-baseline",
            coco_path_to_yolo_path(TWO_STAGE_BASELINE_SHOE_DATASET),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
            "shoe",
        )

        train_and_test_yolo_seg(
            "mugs-two-stage-baseline",
            coco_path_to_yolo_path(TWO_STAGE_BASELINE_MUG_DATASET),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
            "mug",
        )
        train_and_test_yolo_seg(
            "tshirts-two-stage-baseline",
            coco_path_to_yolo_path(TWO_STAGE_BASELINE_TSHIRT_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
            "tshirt",
        )

    def train_on_no_table_random():
        # tshirts, shoes, mugs
        train_and_test_yolo_seg(
            "shoes-no-table-random",
            coco_path_to_yolo_path(RANDOM_TEXTURE_BASELINE_SHOE_NO_TABLE_DATASET),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
            "shoe",
        )

        train_and_test_yolo_seg(
            "mugs-no-table-random",
            coco_path_to_yolo_path(RANDOM_TEXTURE_BASELINE_MUG_NO_TABLE_DATASET),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
            "mug",
        )
        train_and_test_yolo_seg(
            "tshirts-no-table-random",
            coco_path_to_yolo_path(RANDOM_TEXTURE_BASELINE_TSHIRT_NO_TABLE_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
            "tshirt",
        )

    def train_on_no_table_one_stage_diffusion():
        # tshirts, shoes, mugs
        train_and_test_yolo_seg(
            "shoes-no-table-one-stage",
            coco_path_to_yolo_path(ONE_STAGE_NO_TABLE_SHOE_DATASET),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
            "shoe",
        )

        train_and_test_yolo_seg(
            "mugs-no-table-one-stage",
            coco_path_to_yolo_path(ONE_STAGE_NO_TABLE_MUG_DATASET),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
            "mug",
        )
        train_and_test_yolo_seg(
            "tshirts-no-table-one-stage",
            coco_path_to_yolo_path(ONE_STAGE_NO_TABLE_TSHIRT_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
            "tshirt",
        )

    def train_on_three_stage_diffusion():
        # tshirts, shoes, mugs
        train_and_test_yolo_seg(
            "shoes-three-stage",
            coco_path_to_yolo_path(THREE_STAGE_SHOE_DATASET),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
            "shoe",
        )

        train_and_test_yolo_seg(
            "mugs-three-stage",
            coco_path_to_yolo_path(THREE_STAGE_MUG_DATASET),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
            "mug",
        )
        train_and_test_yolo_seg(
            "tshirts-three-stage",
            coco_path_to_yolo_path(THREE_STAGE_TSHIRT_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
            "tshirt",
        )

    def train_on_dual_inpainting():
        # tshirts, shoes, mugs
        train_and_test_yolo_seg(
            "shoes-dual-inpainting",
            coco_path_to_yolo_path(DUAL_INPAINT_SHOES_DATASET),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
            "shoe",
        )

        train_and_test_yolo_seg(
            "mugs-dual-inpainting",
            coco_path_to_yolo_path(DUAL_INPAINT_MUGS_DATASET),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
            "mug",
        )
        train_and_test_yolo_seg(
            "tshirts-dual-inpainting",
            coco_path_to_yolo_path(DUAL_INPAINT_TSHIRTS_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
            "tshirt",
        )

    def train_on_img2img():
        # tshirts, shoes, mugs
        train_and_test_yolo_seg(
            "shoes-img2img",
            coco_path_to_yolo_path(IMG2IMG_SHOES_DATASET),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
            "shoe",
        )

        train_and_test_yolo_seg(
            "mugs-img2img",
            coco_path_to_yolo_path(IMG2IMG_MUGS_DATASET),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
            "mug",
        )
        train_and_test_yolo_seg(
            "tshirts-img2img",
            coco_path_to_yolo_path(IMG2IMG_TSHIRTS_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
            "tshirt",
        )

    def train_on_large_scale_diffusion():
        #   # tshirts, shoes, mugs

        #     # 250

        #     train_and_test_yolo_seg(
        #         "shoes-scale-diffusion-250",
        #         scale_experiment_coco_to_yolo(ONE_STAGE_LARGE_SHOE_DATASET,250),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
        #         "shoe",
        #     )

        #     train_and_test_yolo_seg(
        #         "mugs-scale-diffusion-250",
        #         scale_experiment_coco_to_yolo(ONE_STAGE_LARGE_MUG_DATASET,250),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
        #         "mug",
        #     )

        #     train_and_test_yolo_seg(
        #         "tshirts-scale-diffusion-250",
        #         scale_experiment_coco_to_yolo(ONE_STAGE_LARGE_TSHIRT_DATASET,250),
        #         coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
        #         coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
        #         "tshirt",
        #     )

        #     # 500
        #     train_and_test_yolo_seg(
        #         "shoes-scale-diffusion-500",
        #         scale_experiment_coco_to_yolo(ONE_STAGE_LARGE_SHOE_DATASET,500),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
        #         "shoe",
        #     )

        #     train_and_test_yolo_seg(
        #         "mugs-scale-diffusion-500",
        #         scale_experiment_coco_to_yolo(ONE_STAGE_LARGE_MUG_DATASET,500),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
        #         "mug",
        #     )

        #     train_and_test_yolo_seg(
        #         "tshirts-scale-diffusion-500",
        #         scale_experiment_coco_to_yolo(ONE_STAGE_LARGE_TSHIRT_DATASET,500),
        #         coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
        #         coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
        #         "tshirt",
        #     )

        #     # 1K
        #     train_and_test_yolo_seg(
        #         "shoes-scale-diffusion-1K",
        #         scale_experiment_coco_to_yolo(ONE_STAGE_LARGE_SHOE_DATASET,1000),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
        #         "shoe",
        #     )

        #     train_and_test_yolo_seg(
        #         "mugs-scale-diffusion-1K",
        #         scale_experiment_coco_to_yolo(ONE_STAGE_LARGE_MUG_DATASET,1000),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
        #         "mug",
        #     )
        #     train_and_test_yolo_seg(
        #         "tshirts-scale-diffusion-1K",
        #         scale_experiment_coco_to_yolo(ONE_STAGE_LARGE_TSHIRT_DATASET,1000),
        #         coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
        #         coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
        #         "tshirt",
        #     )
        #     # 2K
        #     train_and_test_yolo_seg(
        #         "shoes-scale-diffusion-2K",
        #         scale_experiment_coco_to_yolo(ONE_STAGE_LARGE_SHOE_DATASET,2000),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
        #         "shoe",
        #     )

        #     train_and_test_yolo_seg(
        #         "mugs-scale-diffusion-2K",
        #         scale_experiment_coco_to_yolo(ONE_STAGE_LARGE_MUG_DATASET,2000),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
        #         "mug",
        #     )

        #     train_and_test_yolo_seg(
        #         "tshirts-scale-diffusion-2K",
        #         scale_experiment_coco_to_yolo(ONE_STAGE_LARGE_TSHIRT_DATASET,2000),
        #         coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
        #         coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
        #         "tshirt",
        #     )

        #     # 5K
        #     train_and_test_yolo_seg(
        #         "shoes-scale-diffusion-5K",
        #         scale_experiment_coco_to_yolo(ONE_STAGE_LARGE_SHOE_DATASET,5000),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
        #         "shoe",
        #     )

        #     train_and_test_yolo_seg(
        #         "mugs-scale-diffusion-5K",
        #         scale_experiment_coco_to_yolo(ONE_STAGE_LARGE_MUG_DATASET,5000),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
        #         "mug",
        #     )

        #     # 10K

        #     train_and_test_yolo_seg(
        #         "shoes-scale-diffusion-10K",
        #         scale_experiment_coco_to_yolo(ONE_STAGE_LARGE_SHOE_DATASET,10000),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
        #         "shoe",
        #     )

        #     train_and_test_yolo_seg(
        #         "mugs-scale-diffusion-10K",
        #         scale_experiment_coco_to_yolo(ONE_STAGE_LARGE_MUG_DATASET,10000),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
        #         coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
        #         "mug",
        #     )

        #     train_and_test_yolo_seg(
        #         "tshirts-scale-diffusion-10K",
        #         scale_experiment_coco_to_yolo(ONE_STAGE_LARGE_TSHIRT_DATASET,10000),
        #         coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
        #         coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
        #         "tshirt",
        #     )
        train_and_test_yolo_seg(
            "tshirts-scale-diffusion-5K",
            scale_experiment_coco_to_yolo(ONE_STAGE_LARGE_TSHIRT_DATASET, 5000),
            coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
            "tshirt",
        )

    def train_on_large_scale_random():

        # 250

        train_and_test_yolo_seg(
            "shoes-scale-random-250",
            scale_experiment_coco_to_yolo(RANDOM_TEXTURE_SHOE_LARGE_DATASET, 250),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
            "shoe",
        )

        train_and_test_yolo_seg(
            "mugs-scale-random-250",
            scale_experiment_coco_to_yolo(RANDOM_TEXTURE_MUG_LARGE_DATASET, 250),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
            "mug",
        )

        train_and_test_yolo_seg(
            "tshirts-scale-random-250",
            scale_experiment_coco_to_yolo(RANDOM_TEXTURE_TSHIRT_LARGE_DATASET, 250),
            coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
            "tshirt",
        )

        # 500
        train_and_test_yolo_seg(
            "shoes-scale-random-500",
            scale_experiment_coco_to_yolo(RANDOM_TEXTURE_SHOE_LARGE_DATASET, 500),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
            "shoe",
        )

        train_and_test_yolo_seg(
            "mugs-scale-random-500",
            scale_experiment_coco_to_yolo(RANDOM_TEXTURE_MUG_LARGE_DATASET, 500),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
            "mug",
        )

        train_and_test_yolo_seg(
            "tshirts-scale-random-500",
            scale_experiment_coco_to_yolo(RANDOM_TEXTURE_TSHIRT_LARGE_DATASET, 500),
            coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
            "tshirt",
        )

        # 1K

        train_and_test_yolo_seg(
            "shoes-scale-random-1K",
            scale_experiment_coco_to_yolo(RANDOM_TEXTURE_SHOE_LARGE_DATASET, 1000),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
            "shoe",
        )

        train_and_test_yolo_seg(
            "mugs-scale-random-1K",
            scale_experiment_coco_to_yolo(RANDOM_TEXTURE_MUG_LARGE_DATASET, 1000),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
            "mug",
        )

        train_and_test_yolo_seg(
            "tshirts-scale-random-1K",
            scale_experiment_coco_to_yolo(RANDOM_TEXTURE_TSHIRT_LARGE_DATASET, 1000),
            coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
            "tshirt",
        )

        # 2K

        train_and_test_yolo_seg(
            "shoes-scale-random-2K",
            scale_experiment_coco_to_yolo(RANDOM_TEXTURE_SHOE_LARGE_DATASET, 2000),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
            "shoe",
        )

        train_and_test_yolo_seg(
            "mugs-scale-random-2K",
            scale_experiment_coco_to_yolo(RANDOM_TEXTURE_MUG_LARGE_DATASET, 2000),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
            "mug",
        )

        train_and_test_yolo_seg(
            "tshirts-scale-random-2K",
            scale_experiment_coco_to_yolo(RANDOM_TEXTURE_TSHIRT_LARGE_DATASET, 2000),
            coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
            "tshirt",
        )

        # 5K

        train_and_test_yolo_seg(
            "shoes-scale-random-5K",
            scale_experiment_coco_to_yolo(RANDOM_TEXTURE_SHOE_LARGE_DATASET, 5000),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
            "shoe",
        )

        train_and_test_yolo_seg(
            "mugs-scale-random-5K",
            scale_experiment_coco_to_yolo(RANDOM_TEXTURE_MUG_LARGE_DATASET, 5000),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
            "mug",
        )

        train_and_test_yolo_seg(
            "tshirts-scale-random-5K",
            scale_experiment_coco_to_yolo(RANDOM_TEXTURE_TSHIRT_LARGE_DATASET, 5000),
            coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
            "tshirt",
        )

        # 10K

        train_and_test_yolo_seg(
            "shoes-scale-random-10K",
            scale_experiment_coco_to_yolo(RANDOM_TEXTURE_SHOE_LARGE_DATASET, 10000),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_SHOES_TEST_DATASET)),
            "shoe",
        )

        train_and_test_yolo_seg(
            "mugs-scale-random-10K",
            scale_experiment_coco_to_yolo(RANDOM_TEXTURE_MUG_LARGE_DATASET, 10000),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_VAL_DATASET)),
            coco_path_to_yolo_path(real_dataset_to_masked(REAL_MUGS_TEST_DATASET)),
            "mug",
        )

        train_and_test_yolo_seg(
            "tshirts-scale-random-10K",
            scale_experiment_coco_to_yolo(RANDOM_TEXTURE_TSHIRT_LARGE_DATASET, 10000),
            coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
            coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
            "tshirt",
        )

    ### TRAIN COMMANDS

    # train_on_real_datasets()
    # train_on_random_texture_baselines()
    # train_on_prompt_experiments()
    # train_on_two_stage_baseline()
    # train_on_no_table_random()
    # train_on_no_table_one_stage_diffusion()
    # train_on_three_stage_diffusion()
    # train_on_dual_inpainting()
    # train_on_img2img()
    train_on_large_scale_diffusion()
    # train_on_large_scale_random()
