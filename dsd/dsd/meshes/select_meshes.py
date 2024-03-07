import glob
import json
import os

import open3d as o3d


def load_keypoints(keypoint_file):
    with open(keypoint_file, "r") as f:
        keypoints = json.load(f)
    return keypoints


def select_meshes(folder_path):
    to_keep = []
    obj_files = glob.glob(os.path.join(folder_path, "**/*.obj"), recursive=True)
    for obj_file in obj_files:
        o3d_mesh = o3d.io.read_triangle_mesh(obj_file)
        # convert to point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d_mesh.vertices

        # apply keypoints
        keypoint_file = obj_file.replace(".obj", "_keypoints.json")
        keypoints = load_keypoints(keypoint_file)
        o3d_kp = [o3d.geometry.TriangleMesh.create_sphere(radius=0.01) for _ in range(len(keypoints))]
        for i, (keypoint_name, coordinates) in enumerate(keypoints.items()):
            o3d_kp[i].translate(coordinates)
        o3d.visualization.draw_geometries([pcd, o3d_mesh] + o3d_kp)
        c = input("Select this mesh? (y/n)")
        if c == "y":
            to_keep.append(obj_file)
        print(f"Selected {len(to_keep)} meshes")
        print(to_keep)
    return to_keep


if __name__ == "__main__":
    import pathlib
    import shutil

    from dsd import DATA_DIR

    # folder_path = DATA_DIR / "meshes" / "objaverse-mugs"
    # keep_list = select_meshes(folder_path)

    keep_list = [
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/4fbb57124fec47d8b216101e19f5d385/4fbb57124fec47d8b216101e19f5d385.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/52d54d64e2544ace8a41d33d2ee03fdc/52d54d64e2544ace8a41d33d2ee03fdc.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/fa7abc1ab50c484db3323ccbf7f8514b/fa7abc1ab50c484db3323ccbf7f8514b.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/922cd7d18c6748d49fe651ded8a04cf4/922cd7d18c6748d49fe651ded8a04cf4.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/58a48a137b03468fac88eecc6dacbaf7/58a48a137b03468fac88eecc6dacbaf7.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/ed42b058fda84754b4a4ce96b17bc91c/ed42b058fda84754b4a4ce96b17bc91c.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/c7f7a78ec67d42819bf33dbf82dd1bbe/c7f7a78ec67d42819bf33dbf82dd1bbe.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/3a2a7c597431416aa7655da8f747424b/3a2a7c597431416aa7655da8f747424b.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/77f00280d56b468381d75bf9392496be/77f00280d56b468381d75bf9392496be.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/5f7108c70d6c482c99ddcba2533c1e87/5f7108c70d6c482c99ddcba2533c1e87.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/12652c6e0aaf44dda32aab816f433baf/12652c6e0aaf44dda32aab816f433baf.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/6142a5564edb4a7ba1c948673460afe0/6142a5564edb4a7ba1c948673460afe0.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/43e95c43931a4801849639bb08ece288/43e95c43931a4801849639bb08ece288.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/91c5283b27c74583900d5e26e2fcd086/91c5283b27c74583900d5e26e2fcd086.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/005349b8ece7424984e8b1f224a8dfe3/005349b8ece7424984e8b1f224a8dfe3.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/32d5033eb7e7435696369ee300ad16e7/32d5033eb7e7435696369ee300ad16e7.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/30f383a4c61d4ddfb6035c931f6639c6/30f383a4c61d4ddfb6035c931f6639c6.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/9dcdde5bf82e4fe2af13784ac2ed0902/9dcdde5bf82e4fe2af13784ac2ed0902.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/1dd7467ee6b947e6b6cd03368c00a8e1/1dd7467ee6b947e6b6cd03368c00a8e1.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/1dd9b373e3084cc0914a580ce5728cf8/1dd9b373e3084cc0914a580ce5728cf8.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/0166cd3012284d0cb89d0c6548f9680c/0166cd3012284d0cb89d0c6548f9680c.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/7cc9e64d7ffd4e369595062abb3ab177/7cc9e64d7ffd4e369595062abb3ab177.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/d6c7ed9bbb2a44e591a8c5168adfe362/d6c7ed9bbb2a44e591a8c5168adfe362.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/3146c02a2b5e41b9ac8481c66c8e1ba1/3146c02a2b5e41b9ac8481c66c8e1ba1.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/3e5b44b8eac24e2dadcc158e7ae421b2/3e5b44b8eac24e2dadcc158e7ae421b2.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/c609327dd3a74fb597584e1b4a14a615/c609327dd3a74fb597584e1b4a14a615.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/7bc2b14a48d1413e9375c32a53e3ee6f/7bc2b14a48d1413e9375c32a53e3ee6f.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/157ab13a82c6409b8ed79ee1073e2227/157ab13a82c6409b8ed79ee1073e2227.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/82e943e3cfac4da5a20d189ddbd6abdf/82e943e3cfac4da5a20d189ddbd6abdf.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/8632380517a043ccbd364e1418cfed74/8632380517a043ccbd364e1418cfed74.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/1522d7b6d8a14f4f8c19f996320fdc09/1522d7b6d8a14f4f8c19f996320fdc09.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/ae7142127dd84ebbbe7762368ace452c/ae7142127dd84ebbbe7762368ace452c.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/97ddd7db0b5b471983db0d6526991dc1/97ddd7db0b5b471983db0d6526991dc1.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/fc63df9772a4474ab070fd765bfbadd9/fc63df9772a4474ab070fd765bfbadd9.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/45381c7f838540d88ed85cd2f00cb35b/45381c7f838540d88ed85cd2f00cb35b.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/6734e6f2fc6e4f80a887e6cfc9c92cc4/6734e6f2fc6e4f80a887e6cfc9c92cc4.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/3a973e2cd37c430a98fd6270687c5c81/3a973e2cd37c430a98fd6270687c5c81.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/50db8b27c1414717b1aed8bd48ecf88d/50db8b27c1414717b1aed8bd48ecf88d.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/c25119c5ac6e4654be3b75d78e34a912/c25119c5ac6e4654be3b75d78e34a912.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/6702d5d92c9c416eac5cc99ee383ff46/6702d5d92c9c416eac5cc99ee383ff46.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/76c92ae8920e4bd4b553122fadc8d570/76c92ae8920e4bd4b553122fadc8d570.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/45ea75466acd45f481ac0fe22beb4aab/45ea75466acd45f481ac0fe22beb4aab.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/9272682c0f714933ac59202500f8d6c8/9272682c0f714933ac59202500f8d6c8.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/826a236ea5fa4c0fbc8c6ee7d88df70d/826a236ea5fa4c0fbc8c6ee7d88df70d.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/d240a7bf160e48418c2f16e91cd62987/d240a7bf160e48418c2f16e91cd62987.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/49a6fe4995224d4e8e5ee1e23727e51d/49a6fe4995224d4e8e5ee1e23727e51d.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/b8478cea51454555b90de0fe6ba7ba83/b8478cea51454555b90de0fe6ba7ba83.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/2a8632e9355948ad97c129f957c2ad09/2a8632e9355948ad97c129f957c2ad09.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/92dafaefb8c742fb95fb8ec71c27b2b5/92dafaefb8c742fb95fb8ec71c27b2b5.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/047462f5fdc244339511f87f4df49446/047462f5fdc244339511f87f4df49446.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/44b45ae8126a4358ac49321c538e2c31/44b45ae8126a4358ac49321c538e2c31.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/721c70b175674aad93d742ff8812b512/721c70b175674aad93d742ff8812b512.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/057ed726c32c4e0d8912d04343e7bf5a/057ed726c32c4e0d8912d04343e7bf5a.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/db9345f568e8499a9eac2577302b5f51/db9345f568e8499a9eac2577302b5f51.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/3576b3e274ea486f8ffeeead0259ea8e/3576b3e274ea486f8ffeeead0259ea8e.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/93ad19db34bc49a0b7acc24e7a5a22fd/93ad19db34bc49a0b7acc24e7a5a22fd.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/2555a69678e64f91a286dd8132f7e937/2555a69678e64f91a286dd8132f7e937.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/ce6383d8fb27451eb8c39a7193194307/ce6383d8fb27451eb8c39a7193194307.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/44d69f20e761400d93b0f1e72fe08528/44d69f20e761400d93b0f1e72fe08528.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/6543aaad61124f99a2296099a7f56748/6543aaad61124f99a2296099a7f56748.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/9de0f410e48341bc83b7ed43a864e34b/9de0f410e48341bc83b7ed43a864e34b.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/2b3e80b25a434b15b31cd5078b7f059a/2b3e80b25a434b15b31cd5078b7f059a.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/c38d573bb47f4dc4bdd2faea926c5276/c38d573bb47f4dc4bdd2faea926c5276.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/f173eedc45904558824415d09646753e/f173eedc45904558824415d09646753e.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/55b689d55dbb45dc8e4a5d7d76e54daa/55b689d55dbb45dc8e4a5d7d76e54daa.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/1f6a331e21db4f919f6e448bfde3a090/1f6a331e21db4f919f6e448bfde3a090.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/e2b0bc9174b3467da5ffaeddc44fca8f/e2b0bc9174b3467da5ffaeddc44fca8f.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/77d41aba53e4455f9f84fa04b175dff4/77d41aba53e4455f9f84fa04b175dff4.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/92e8a44f3259460d99de2e6d20b00606/92e8a44f3259460d99de2e6d20b00606.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/6e74ee87e8a94e70ac369dbb2d53b0ae/6e74ee87e8a94e70ac369dbb2d53b0ae.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/c666d90ad57e44f1baf90e5ec062a152/c666d90ad57e44f1baf90e5ec062a152.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/4fbbdf9506be4b6ea0d4ce74007a37f8/4fbbdf9506be4b6ea0d4ce74007a37f8.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/7ae72d29ede943798623e11231b70109/7ae72d29ede943798623e11231b70109.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/63660b72eff64e63947bd7fd84629d94/63660b72eff64e63947bd7fd84629d94.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/ad2546804e8845b09e5db7c6db16b704/ad2546804e8845b09e5db7c6db16b704.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/3a87a55210db4fd9a27565671824f894/3a87a55210db4fd9a27565671824f894.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/bcd8eaf8dc13433493944a7f6b882f9a/bcd8eaf8dc13433493944a7f6b882f9a.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/2c88306f3d724a839054f3c2913fb1d5/2c88306f3d724a839054f3c2913fb1d5.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/86a4e42f3f0e4847b58413ab69f0c789/86a4e42f3f0e4847b58413ab69f0c789.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/74f2d58ffb5f4fc8951be72cd13f64d6/74f2d58ffb5f4fc8951be72cd13f64d6.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/96d0e25b866c48dbb7c90d26298c91e1/96d0e25b866c48dbb7c90d26298c91e1.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/435077bef9dd48d6a5cee72dd1f4a4ef/435077bef9dd48d6a5cee72dd1f4a4ef.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/ca4f9a92cc2f4ee98fe9332db41bf7f7/ca4f9a92cc2f4ee98fe9332db41bf7f7.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/9321c45cb9cf459f9f803507d3a11fb3/9321c45cb9cf459f9f803507d3a11fb3.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/89591a21a3ad499f9178247b1ea78484/89591a21a3ad499f9178247b1ea78484.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/95fe6a8daebb4ea5846ace8c0806d108/95fe6a8daebb4ea5846ace8c0806d108.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/2ce176f569cf41709629e713ece8d1bb/2ce176f569cf41709629e713ece8d1bb.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/d40aa57bd3134878b573ba48cfa37f2f/d40aa57bd3134878b573ba48cfa37f2f.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/a17de88a9464410fa2c8f550ab04749e/a17de88a9464410fa2c8f550ab04749e.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/b65cba53888f46a4b3c363f1836d0851/b65cba53888f46a4b3c363f1836d0851.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/13bd7511918640fba71dd4ca554d5a5e/13bd7511918640fba71dd4ca554d5a5e.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/3ea2fd4e065147048064f4c97a89fe6f/3ea2fd4e065147048064f4c97a89fe6f.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/b4e5f0e6367e442c995eb1e241e61f74/b4e5f0e6367e442c995eb1e241e61f74.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/a30aab65d235438bbfc6699faaea762d/a30aab65d235438bbfc6699faaea762d.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/2e6d9e77be114535b01d7ef028c2f49c/2e6d9e77be114535b01d7ef028c2f49c.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/38e53299cce04fe19235b8b198f7818b/38e53299cce04fe19235b8b198f7818b.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/783d2d80f163497aa02fda539e82755e/783d2d80f163497aa02fda539e82755e.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/fe27c397cf42404ea4c1f6a32a38a27f/fe27c397cf42404ea4c1f6a32a38a27f.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/84f617a1db9d48bb8521a78b6c80b8a1/84f617a1db9d48bb8521a78b6c80b8a1.obj",
        "/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data/meshes/objaverse-mugs/818d7994915d447bbf05be34a0b122fe/818d7994915d447bbf05be34a0b122fe.obj",
    ]
    keep_list = [pathlib.Path(x) for x in keep_list]

    target_dir = DATA_DIR / "meshes" / "objaverse-mugs-filtered"

    relative_keep_list = [x.relative_to(keep_list[0].parents[1]) for x in keep_list]
    print(keep_list)

    for i in range(len(keep_list)):
        target = target_dir / relative_keep_list[i].parents[0]
        target.mkdir(parents=True, exist_ok=True)
        print(f"Copying {keep_list[i]} to {target_dir / relative_keep_list[i]}")
        shutil.copy(keep_list[i], target_dir / relative_keep_list[i])
        kp_path = str(keep_list[i]).replace(".obj", "_keypoints.json")
        shutil.copy(kp_path, target_dir / str(relative_keep_list[i]).replace(".obj", "_keypoints.json"))
