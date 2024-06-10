from pathlib import Path

from airo_dataset_tools.coco_tools.coco_instances_to_yolo import create_yolo_dataset_from_coco_instances_dataset
from paths import DATA_DIR, REAL_DATA_DIR


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


if __name__ == "__main__":
    # create list of all global variables that contain 'TSHIRT' and 'DATASET'
    tshirt_datasets = {k: v for k, v in globals().items() if "TSHIRT" in k and "DATASET" in k}
    print(tshirt_datasets)

    for k, v in tshirt_datasets.items():
        print(k, v)
        # create yolo dataset
        assert isinstance(v, Path)
        yolo_target_path = coco_path_to_yolo_path(v)

        if yolo_target_path.exists():
            print(f"{yolo_target_path} already exists")
            continue
        yolo_target_path.mkdir(parents=True, exist_ok=False)
        create_yolo_dataset_from_coco_instances_dataset(str(v), yolo_target_path, True)
