from pathlib import Path

from airo_dataset_tools.coco_tools.coco_instances_to_yolo import create_yolo_dataset_from_coco_instances_dataset
from paths import (  # noqa
    CCS_COMPARISON_05_MUG_DATASET,
    CCS_COMPARISON_05_SHOE_DATASET,
    CCS_COMPARISON_05_TSHIRT_DATASET,
    CCS_COMPARISON_10_MUG_DATASET,
    CCS_COMPARISON_10_SHOE_DATASET,
    CCS_COMPARISON_10_TSHIRT_DATASET,
    CCS_COMPARISON_15_MUG_DATASET,
    CCS_COMPARISON_15_SHOE_DATASET,
    CCS_COMPARISON_15_TSHIRT_DATASET,
    CCS_COMPARISON_20_MUG_DATASET,
    CCS_COMPARISON_20_SHOE_DATASET,
    CCS_COMPARISON_20_TSHIRT_DATASET,
    CCS_COMPARISON_25_MUG_DATASET,
    CCS_COMPARISON_25_SHOE_DATASET,
    CCS_COMPARISON_25_TSHIRT_DATASET,
    DATA_DIR,
    DUAL_INPAINT_MUGS_DATASET,
    DUAL_INPAINT_SHOES_DATASET,
    DUAL_INPAINT_TSHIRTS_DATASET,
    IMG2IMG_MUGS_DATASET,
    IMG2IMG_SHOES_DATASET,
    IMG2IMG_TSHIRTS_DATASET,
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
    REAL_DATA_DIR,
    REAL_MUGS_TEST_DATASET,
    REAL_MUGS_TRAIN_DATASET,
    REAL_MUGS_VAL_DATASET,
    REAL_SHOES_TEST_DATASET,
    REAL_SHOES_TRAIN_DATASET,
    REAL_SHOES_VAL_DATASET,
    REAL_TSHIRTS_TEST_DATASET,
    REAL_TSHIRTS_TRAIN_DATASET,
    REAL_TSHIRTS_VAL_DATASET,
    TWO_STAGE_BASELINE_MUG_DATASET,
    TWO_STAGE_BASELINE_SHOE_DATASET,
    TWO_STAGE_BASELINE_TSHIRT_DATASET,
)


def coco_path_to_yolo_path(v):
    if "real" in str(v).lower():
        # find path relative to real data dir
        yolo_target_path = v.relative_to(REAL_DATA_DIR)

        # add yolo dir to path
        yolo_target_path = DATA_DIR / "yolo" / yolo_target_path

        # get the filename
        yolo_target_path = yolo_target_path.parent / yolo_target_path.name.split(".")[0]

    else:
        yolo_target_path = Path(str(v).replace("coco", "yolo")).parent
    return yolo_target_path


# todo: document hacky code
import pathlib


def scale_experiment_coco_to_yolo(coco_path, size):
    coco_path = str(coco_path)
    yolo_path = pathlib.Path(coco_path.replace("coco", "yolo")).parents[0] / f"{size}"
    return yolo_path


if __name__ == "__main__":

    def generate_datasets(dataset_dict):
        for k, v in dataset_dict.items():
            print(k, v)
            # create yolo dataset
            assert isinstance(v, Path)
            yolo_target_path = coco_path_to_yolo_path(v)

            if yolo_target_path.exists():
                print(f"{yolo_target_path} already exists")
                continue
            yolo_target_path.mkdir(parents=True, exist_ok=False)
            create_yolo_dataset_from_coco_instances_dataset(str(v), str(yolo_target_path), True)

    # create list of all global variables that contain 'TSHIRT' and 'DATASET'
    tshirt_datasets = {k: v for k, v in globals().items() if "TSHIRT" in k and "DATASET" in k}
    print(tshirt_datasets)
    generate_datasets(tshirt_datasets)

    # create list of all global variables that contain 'SHOE' and 'DATASET'
    shoe_datasets = {k: v for k, v in globals().items() if "SHOE" in k and "DATASET" in k}
    for k, v in shoe_datasets.items():
        if "REAL" in k:
            shoe_datasets[k] = Path(str(v).replace(".json", "_sam_masked.json"))
    print(shoe_datasets)
    generate_datasets(shoe_datasets)

    # create list of all global variables that contain 'MUG' and 'DATASET'
    mug_datasets = {k: v for k, v in globals().items() if "MUG" in k and "DATASET" in k}
    for k, v in mug_datasets.items():
        if "REAL" in k:
            mug_datasets[k] = Path(str(v).replace(".json", "_sam_masked.json"))
    print(mug_datasets)
    generate_datasets(mug_datasets)

    # scale dataset experiments

    from paths import (
        ONE_STAGE_LARGE_MUG_DATASET,
        ONE_STAGE_LARGE_SHOE_DATASET,
        ONE_STAGE_LARGE_TSHIRT_DATASET,
        RANDOM_TEXTURE_MUG_LARGE_DATASET,
        RANDOM_TEXTURE_SHOE_LARGE_DATASET,
        RANDOM_TEXTURE_TSHIRT_LARGE_DATASET,
    )

    large_scale_datasets = (
        ONE_STAGE_LARGE_MUG_DATASET,
        ONE_STAGE_LARGE_SHOE_DATASET,
        ONE_STAGE_LARGE_TSHIRT_DATASET,
        RANDOM_TEXTURE_MUG_LARGE_DATASET,
        RANDOM_TEXTURE_TSHIRT_LARGE_DATASET,
        RANDOM_TEXTURE_SHOE_LARGE_DATASET,
    )
    scale_sizes = (250, 500, 1000, 2000, 5000, 10000)

    import itertools

    for (size, dataset) in itertools.product(scale_sizes, large_scale_datasets):
        # get the annotations path

        coco_path = f"{str(pathlib.Path(dataset).with_suffix(''))}_{size}.json"

        # determine a yolo path
        yolo_path = scale_experiment_coco_to_yolo(coco_path, size)

        # convert
        print(f"{coco_path} -> \n{yolo_path}")

        create_yolo_dataset_from_coco_instances_dataset(str(coco_path), str(yolo_path), True)

    ### visulize in fifyftone
    # import fiftyone as fo

    # name = "my-dataset"
    # data_path = coco_path_to_yolo_path(shoe_datasets["REAL_SHOES_VAL_DATASET"])
    # classes = ["shoe"]
    # print(data_path)

    # # Import dataset by explicitly providing paths to the source media and labels
    # dataset = fo.Dataset.from_dir(
    #     dataset_type=fo.types.YOLOv5Dataset,
    #     dataset_dir=data_path,
    #     label_type="polylines",
    # )

    # # view the dataset

    # session = fo.launch_app(dataset)
    # session.view = dataset.take(100)  # limit the view to the first 10 samples
    # # keep the session running
    # session.wait()
