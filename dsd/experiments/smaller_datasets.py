import copy
import json
import pathlib
from random import sample

from airo_dataset_tools.data_parsers.coco import CocoInstancesDataset


def create_split_datasets(dataset_json_path, split_sizes):
    coco_dict = CocoInstancesDataset(**json.load(open(dataset_json_path, "r")))
    for split_size in split_sizes:
        assert len(coco_dict.annotations) >= split_size, f"{split_size} > {len(coco_dict.annotations)}"

        new_coco_dict = copy.deepcopy(coco_dict)
        new_coco_dict.annotations = sample(new_coco_dict.annotations, split_size)
        new_coco_filename = f"{str(pathlib.Path(dataset_json_path).with_suffix(''))}_{split_size}.json"
        print(new_coco_filename)
        with open(new_coco_filename, "w") as file:
            json.dump(new_coco_dict.model_dump(), file)


if __name__ == "__main__":
    file_path = "/home/tlips/Documents/diffusing-synthetic-data/data/coco/mugs/random_textures_large/random_textures/random_textures/annotations.json"
    split_sizes = [1000, 2000, 5000, 10000]
    create_split_datasets(file_path, split_sizes)
