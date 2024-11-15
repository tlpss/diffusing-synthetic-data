GSO_DIR = "/home/tlips/Documents/blender-assets/GSO/"
import shutil

target_dir = "/home/tlips/Documents/diffusing-synthetic-data/data/meshes/shoes/GSO"

import pathlib


class Metadata:
    pass


model_metadata = Metadata()


mesh_dirs = [x for x in pathlib.Path(GSO_DIR).iterdir() if x.is_dir()]

shoe_mesh_dirs = []

for mesh_dir in mesh_dirs:
    print(mesh_dir)
    protobuf_json = mesh_dir / "metadata.pbtxt"
    with open(protobuf_json, "r") as f:
        metadata = f.readlines()
        metadata = "".join(metadata)

        if "Shoe" in metadata:
            print(mesh_dir)
            shoe_mesh_dirs.append(mesh_dir)


target_dir = pathlib.Path(target_dir)
target_dir.mkdir(exist_ok=True)

for shoe_mesh_dir in shoe_mesh_dirs:
    # copy mesh to the target dir
    print(shoe_mesh_dir / "meshes" / "model.obj")
    target_filename = shoe_mesh_dir.stem + ".obj"
    target_filename = target_dir / target_filename
    print(target_filename)
    shutil.copy(shoe_mesh_dir / "meshes" / "model.obj", target_filename)
