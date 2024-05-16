import pathlib

import bpy

data_dir = "/home/tlips/Documents/synthetic-cloth-data/synthetic-cloth-data/data/deformed_meshes/TSHIRT/dsd-tshirts"
target_dir = (
    "/home/tlips/Documents/synthetic-cloth-data/synthetic-cloth-data/data/deformed_meshes/TSHIRT/dsd-tshirts-b"
)
target_dir = pathlib.Path(target_dir)
target_dir.mkdir(parents=True, exist_ok=True)

meshes = pathlib.Path(data_dir).rglob("*.obj")
meshes = list(meshes)

# convert from y-up to z-up
for mesh in meshes:
    bpy.ops.import_scene.obj(filepath=str(mesh), split_mode="OFF")
    bpy.ops.transform.rotate(value=-1.5708, orient_axis="X", orient_type="GLOBAL")
    # apply rotation
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    bpy.ops.export_scene.obj(filepath=str(target_dir / mesh.name), use_selection=True)
    bpy.ops.object.delete()
