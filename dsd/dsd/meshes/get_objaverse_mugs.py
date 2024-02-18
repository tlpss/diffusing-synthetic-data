import os
import pathlib

import numpy as np
import objaverse
import trimesh
from tqdm import tqdm

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


def download_objaverse_objects_as_obj(uids, directory: str):
    directory = pathlib.Path(directory)
    os.makedirs(directory, exist_ok=True)
    object_dict = objaverse.load_objects(uids)
    for uuid, mesh_path in object_dict.items():
        print(f"converting {mesh_path} to {directory}")
        mesh = trimesh.load(mesh_path)
        mesh_dir = DATA_DIR / "meshes" / directory / f"{uuid}/"
        mesh_dir.mkdir(exist_ok=True, parents=True)
        try:
            mesh.export(mesh_dir / f"{uuid}.obj")
        except AttributeError:
            print(f"Error exporting {uuid}")


def gather_objaverse_mug_tags():
    lvis_annotations = objaverse.load_lvis_annotations()
    mug_uids = lvis_annotations["mug"]
    return mug_uids


def normalize_meshes(directory: str):
    directory = pathlib.Path(directory)
    meshes = list(directory.glob("**/*.obj"))
    for mesh in tqdm(meshes):
        normalize_mesh(mesh)


def normalize_mesh(obj_path):

    # force mesh https://github.com/mikedh/trimesh/issues/1585
    mesh = trimesh.load(obj_path, force="mesh")

    # remesh to 10K vertices
    mesh = mesh.simplify_quadric_decimation(5000)

    # from y up to z up
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))

    # center the object
    mesh.vertices -= mesh.center_mass
    mesh.vertices[:, 2] -= mesh.bounds[0][2]  # align bottom with z=0

    # scale y axis to 10cm because the meshes  have no uniform measure scale.
    scale_factor = 0.1 / mesh.extents[1]
    mesh.vertices *= scale_factor

    # remove target file if it exists
    if obj_path.exists():
        obj_path.unlink()
    mesh.export(obj_path.parent / obj_path)


if __name__ == "__main__":
    from dsd import DATA_DIR

    mug_uids = gather_objaverse_mug_tags()
    download_objaverse_objects_as_obj(mug_uids, DATA_DIR / "meshes" / "objaverse-mugs")
    normalize_meshes(DATA_DIR / "meshes" / "objaverse-mugs")

    # remove the following meshes because they are no good (manual selection)
    REMOVE_MESHES = [
        "85f1874605a44d239e11fefa2e548cb2",
    ]
    for mesh in REMOVE_MESHES:
        (DATA_DIR / "meshes" / "objaverse-mugs" / mesh).rmdir()
