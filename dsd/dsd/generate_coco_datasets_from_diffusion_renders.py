"script to create coco annotations for diffusion rendered images"
import json
import pathlib
from collections import defaultdict

import cv2
import numpy as np
import tqdm
from airo_dataset_tools.data_parsers.coco import (
    CocoImage,
    CocoKeypointAnnotation,
    CocoKeypointCategory,
    CocoKeypointsDataset,
)
from airo_dataset_tools.segmentation_mask_converter import BinarySegmentationMask
from PIL import Image


def get_images_per_renderer(render_path: pathlib.Path):
    # get all files in the folder
    renderer_to_images_dict = defaultdict(list)
    image_paths = list(render_path.glob("**/*.png"))
    image_paths.extend(list(render_path.glob("**/*.jpg")))
    # remove all images with "rgb.png", "depth_image.png" ,"segmentation.png"
    image_paths = [x for x in image_paths if not str(x.parents[0]).endswith("original")]

    for path in image_paths:
        # check if image is all black -> NSFW filter -> skip
        image = cv2.imread(str(path))
        if np.max(image) < 1e-4:
            continue

        renderer = path.parent.relative_to(path.parents[1])
        renderer_to_images_dict[str(renderer)].append(path)
    return renderer_to_images_dict


def get_annotations_for_rendered_image(image_path, coco_category):
    original_dir_path = image_path.parents[1] / "original"
    mask_path = original_dir_path / "segmentation.png"
    keypoints_path = original_dir_path / "keypoints.json"

    mask = np.array(Image.open(mask_path))
    mask = mask > 0.5
    mask = BinarySegmentationMask(mask)

    keypoint_dict = json.load(open(keypoints_path))
    keypoint_list = []
    n_keypoints = 0

    for name in coco_category.keypoints:
        keypoint = keypoint_dict[name]
        keypoint_list.extend(keypoint)
        n_keypoints += 1 * (keypoint[2] > 0.5)

    annotation = CocoKeypointAnnotation(
        id=0,
        image_id=0,
        category_id=0,
        bbox=mask.bbox,
        segmentation=mask.as_compressed_rle,
        area=mask.area,
        keypoints=keypoint_list,
        num_keypoints=n_keypoints,
    )
    return annotation


def create_coco_dataset(images, coco_category):
    coco_categories = [coco_category]
    coco_images = []
    coco_annotations = []

    for idx, image_path in enumerate(images):
        image = Image.open(image_path)
        coco_image = CocoImage(id=idx, width=image.width, height=image.height, file_name=str(image_path))
        annotation = get_annotations_for_rendered_image(image_path, coco_category)
        annotation.image_id = idx
        annotation.id = idx
        coco_images.append(coco_image)
        coco_annotations.append(annotation)

    coco_dataset = CocoKeypointsDataset(categories=coco_categories, annotations=coco_annotations, images=coco_images)
    return coco_dataset


def copy_img(src, dst):
    cv2_image = cv2.imread(str(src))
    cv2.imwrite(str(dst), cv2_image)


def copy_images_and_make_paths_relative(
    coco_dataset: CocoKeypointsDataset, render_path: pathlib.Path, coco_dataset_dir: pathlib.Path
):
    image_dataset_dir = coco_dataset_dir / "images"

    copy_task_list = []
    for image in coco_dataset.images:
        abs_path = image.file_name
        abs_path = pathlib.Path(abs_path)
        relative_path = abs_path.relative_to(render_path)

        new_path = image_dataset_dir / relative_path
        new_path.parent.mkdir(parents=True, exist_ok=True)
        image.file_name = str(new_path.relative_to(coco_dataset_dir))
        copy_task_list.append((abs_path, new_path))

    # pool = mp.Pool(mp.cpu_count() - 4)
    # pool.starmap(copy_img, copy_task_list)
    # pool.close()
    # pool.join()
    for src, dst in tqdm.tqdm(copy_task_list):
        copy_img(src, dst)

    return coco_dataset


def generate_coco_datasets(target_coco_path, render_path: pathlib.Path, coco_category: CocoKeypointCategory):
    renderer_to_images_dict = get_images_per_renderer(render_path)
    print("gathered index")
    for renderer, images in renderer_to_images_dict.items():
        coco_dataset = create_coco_dataset(images, coco_category)
        coco_dataset_dir = target_coco_path / renderer
        coco_dataset_dir.mkdir(parents=True, exist_ok=True)

        print("copying images")
        coco_dataset = copy_images_and_make_paths_relative(coco_dataset, render_path, coco_dataset_dir)

        with open(coco_dataset_dir / "annotations.json", "w") as f:
            json.dump(coco_dataset.model_dump(), f)


mug_category = CocoKeypointCategory(  # noqa
    id=0, name="mug", supercategory="mug", keypoints=["bottom", "handle", "top"], skeleton=[[0, 1], [1, 2]]
)

shoe_category = CocoKeypointCategory(  # noqa
    id=0, name="shoe", supercategory="shoe", keypoints=["nose", "heel", "top"], skeleton=[[0, 1], [1, 2]]
)

tshirt_category = CocoKeypointCategory(  # noqa
    id=0,
    name="tshirt",
    supercategory="tshirt",
    keypoints=[
        "shoulder_left",
        "neck_left",
        "neck_right",
        "shoulder_right",
        "sleeve_right_top",
        "sleeve_right_bottom",
        "armpit_right",
        "waist_right",
        "waist_left",
        "armpit_left",
        "sleeve_left_bottom",
        "sleeve_left_top",
    ],
    skeleton=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]],
)


if __name__ == "__main__":

    import click

    @click.command()
    @click.option("--target_coco_path", type=str, required=True)
    @click.option("--render_path", type=str, required=True)
    @click.option("--category", type=str, required=True)
    def generate_coco_datasets_cli(target_coco_path, render_path, category):
        target_coco_path = pathlib.Path(target_coco_path)
        render_path = pathlib.Path(render_path)
        category = eval(category)

        generate_coco_datasets(target_coco_path, render_path, category)

    generate_coco_datasets_cli()
