import numpy as np
import open3d as o3d


def get_mug_keypoints(vertices: np.ndarray):
    margin = 0.005

    bottom_vertices = vertices[vertices[:, 2] < margin]
    bottom_center = bottom_vertices.mean(axis=0)

    bottom_kp = np.concatenate([bottom_center, np.array([0.0])])

    # most outwards point is handle keypoint
    vertices_radius = np.linalg.norm(vertices[..., :2], axis=1)
    handle_kp = vertices[np.argmax(vertices_radius)]

    top_z = np.max(vertices[:, 2])
    top_points = vertices[vertices[:, 2] > top_z - margin]
    # find top point furthest away from handle
    top_kp = top_points[np.argmax(np.linalg.norm(top_points - handle_kp, axis=1))]

    return {"bottom": bottom_kp.tolist(), "handle": handle_kp.tolist(), "top": top_kp.tolist()}


if __name__ == "__main__":
    import json

    from dsd import DATA_DIR

    mug_mesh_path = DATA_DIR / "meshes" / "mugs"
    mug_meshes = list(mug_mesh_path.glob("*.obj"))

    for mug_mesh_path in mug_meshes:
        mesh = o3d.io.read_triangle_mesh(str(mug_mesh_path))
        vertices = np.array(mesh.vertices)
        keypoints = get_mug_keypoints(vertices)

        kp_path = str(mug_mesh_path)[:-4] + "_keypoints.json"
        print(kp_path)
        json.dump(keypoints, open(kp_path, "w"))
