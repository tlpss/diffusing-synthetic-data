import json

from dsd import DATA_DIR

# scenes
SHOE_SCENES_DIR = DATA_DIR / "scenes" / "shoes" / "gso-filtered-2500"
MUG_SCENES_DIR = DATA_DIR / "scenes" / "mugs" / "objaverse-filtered-2500-ral"
TSHIRT_SCENES_DIR = DATA_DIR / "scenes" / "tshirts" / "syncloth-filtered-2500"

# real train  datasets
REAL_DATA_DIR = DATA_DIR / "real"
REAL_SHOES_TRAIN_DATASET = REAL_DATA_DIR / "shoes" / "dsd-shoes-robot-kpcentercropped" / "annotations_train.json"
REAL_MUGS_TRAIN_DATASET = REAL_DATA_DIR / "mugs" / "dsd-mugs-robot" / "annotations_train.json"
REAL_TSHIRTS_TRAIN_DATASET = REAL_DATA_DIR / "tshirts" / "tshirts-train.val-kpcentercropped" / "annotations_train.json"

# eval datasets
REAL_SHOES_VAL_DATASET = REAL_DATA_DIR / "shoes" / "dsd-shoes-robot-kpcentercropped" / "annotations_val.json"
REAL_SHOES_TEST_DATASET = REAL_DATA_DIR / "shoes" / "dsd-shoes-real_resized_512x512" / "annotations.json"


REAL_MUGS_VAL_DATASET = REAL_DATA_DIR / "mugs" / "dsd-mugs-robot" / "annotations_val.json"
REAL_MUGS_TEST_DATASET = REAL_DATA_DIR / "mugs" / "lab-mugs_resized_512x512" / "lab-mugs.json"

REAL_TSHIRTS_VAL_DATASET = REAL_DATA_DIR / "tshirts" / "tshirts-train.val-kpcentercropped" / "annotations_val.json"
REAL_TSHIRTS_TEST_DATASET = REAL_DATA_DIR / "tshirts" / "tshirts-test-kpcentercropped" / "annotations.json"

# random texture baselines
RANDOM_TEXTURE_BASELINE_DIR = DATA_DIR / "random_textures"
SHOE_RANDOM_TEXTURE_RENDER_DIR = RANDOM_TEXTURE_BASELINE_DIR / "shoes" / "001"
MUG_RANDOM_TEXTURE_RENDER_DIR = RANDOM_TEXTURE_BASELINE_DIR / "mugs" / "001"
TSHIRT_RANDOM_TEXTURE_RENDER_DIR = RANDOM_TEXTURE_BASELINE_DIR / "tshirts" / "001"


if __name__ == "__main__":
    dataset_dict = {
        "shoes": {
            "train": REAL_SHOES_TRAIN_DATASET,
            "val": REAL_SHOES_VAL_DATASET,
            "test": REAL_SHOES_TEST_DATASET,
        },
        "mugs": {
            "train": REAL_MUGS_TRAIN_DATASET,
            "val": REAL_MUGS_VAL_DATASET,
            "test": REAL_MUGS_TEST_DATASET,
        },
        "tshirts": {
            "train": REAL_TSHIRTS_TRAIN_DATASET,
            "val": REAL_TSHIRTS_VAL_DATASET,
            "test": REAL_TSHIRTS_TEST_DATASET,
        },
    }

    for k, v in dataset_dict.items():
        print(k)
        for k2, v2 in v.items():
            # open json file
            with open(v2, "r") as f:
                data = json.load(f)
                num_images = len(data["images"])
                print(f"{k}-{k2}    num images: {num_images}")