import os

import objaverse

MUG_UIDS = [
    "ffd1e3be72e34ced8ab5e4251f87b6a2",
    "fa17099f18804409bc6d9e8e397b4681",
    "ef3144db2ffa465182f057060e760cf5",
    "e3cae64e7d3048049b4ea3c9dcb88e2f",
    "b7c302e255f24f8e9bba8c4196396f70",
]


def download_mugs(directory: str):

    os.makedirs(directory, exist_ok=True)
    object_dict = objaverse.load_objects(MUG_UIDS)
    for uid, obj_path in object_dict.items():
        target_obj_path = os.path.join(directory, uid + ".glb")
        print(f"Copying {obj_path} to {target_obj_path}")
        os.system(f"cp {obj_path} {target_obj_path}")


if __name__ == "__main__":
    from dsd import DATA_DIR

    download_mugs(DATA_DIR / "meshes" / "mugs")
