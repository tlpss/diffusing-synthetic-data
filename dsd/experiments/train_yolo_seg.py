from pathlib import Path

import wandb
from ultralytics import YOLO, settings

from dsd import DATA_DIR

settings.update({"datasets_dir": str(DATA_DIR / "yolo")})
# rundir
settings.update({"runs_dir": str(DATA_DIR.parent / "yolo-logs")})
settings.update({"weights_dir": str(DATA_DIR.parent / "yolo-logs")})


# create temp yolo data.yaml file


def create_yolo_seg_data_yaml(train_dataset_path, val_dataset_path, class_name):

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

    with open("data.yaml", "w") as f:
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

    create_yolo_seg_data_yaml(train_dataset, val_dataset, category)

    model.train(data="data.yaml", epochs=50, imgsz=512, name=yolo_train_name, batch=16)

    # evaluate the model
    # load best checkpoint
    model = YOLO(f"{DATA_DIR.parent}/yolo-logs/segment/{yolo_train_name}/weights/best.pt")

    # set test dataset as val to evaluate
    create_yolo_seg_data_yaml(train_dataset, test_dataset, category)
    test_results = model.val(data="data.yaml")
    all_aps = test_results.seg.all_ap
    m_ap = test_results.seg.map

    if wandb.run:
        wandb.log({"test/mAP": m_ap})

    print(f"mAP: {m_ap}")
    print("all APs")

    thresholds = ["0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95"]
    for i, ap in enumerate(all_aps[0]):
        wandb.log({f"test/AP_{thresholds[i]}": ap})

    # remove the temp yolo data.yaml file
    import os

    os.remove("data.yaml")

    WANDB_FINISH()


if __name__ == "__main__":
    from generate_yolo_datasets import coco_path_to_yolo_path
    from paths import RANDOM_TEXTURE_BASELINE_TSHIRT_DATASET, REAL_TSHIRTS_TEST_DATASET, REAL_TSHIRTS_VAL_DATASET

    train_and_test_yolo_seg(
        "tshirts-random",
        coco_path_to_yolo_path(RANDOM_TEXTURE_BASELINE_TSHIRT_DATASET),
        coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET),
        coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET),
        "tshirt",
    )
    # train_and_test_yolo_seg("tshirts-real", coco_path_to_yolo_path(REAL_TSHIRTS_TRAIN_DATASET), coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET), coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET), "tshirt")
    # train_and_test_yolo_seg("tshirts-prompts-blip", coco_path_to_yolo_path(PROMPTS_BLIP_TSHIRT_DATASET), coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET), coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET), "tshirt")
    # train_and_test_yolo_seg("tshirts-prompts-gemini", coco_path_to_yolo_path(PROMPTS_GEMINI_TSHIRT_DATASET), coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET), coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET), "tshirt")
    # train_and_test_yolo_seg("tshirts-prompts-classname", coco_path_to_yolo_path(PROMPTS_CLASSNAME_TSHIRT_DATASET), coco_path_to_yolo_path(REAL_TSHIRTS_VAL_DATASET), coco_path_to_yolo_path(REAL_TSHIRTS_TEST_DATASET), "tshirt")
