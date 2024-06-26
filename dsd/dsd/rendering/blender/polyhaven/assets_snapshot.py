"""script to create asset snapshots (json files with metadata about the assets). These are used to sample assets for the synthetic images in a reproducible way
(needed since assets can be added or removed online, especially on Polyhaven).

Usage:
blender -P -b <path_to_this_file>

Make sure to add (and create if required) the asset files first
"""
import json

import airo_blender as ab

from dsd.rendering.blender.polyhaven import POLYHAVEN_ASSETS_SNAPSHOT_PATH


def create_asset_json(assets, snapshot_path):
    asset_snapshot = {"assets": assets}

    with open(snapshot_path, "w") as file:
        json.dump(asset_snapshot, file, indent=4)


if __name__ == "__main__":
    all_assets = ab.available_assets()

    polyhaven_assets = [asset for asset in all_assets if asset["library"] == "Poly Haven"]
    print(f"Found {len(polyhaven_assets)} polyhaven assets")
    create_asset_json(polyhaven_assets, POLYHAVEN_ASSETS_SNAPSHOT_PATH)

    print("Done!")
