import pathlib

POLYHAVEN_ASSETS_SNAPSHOT_PATH = pathlib.Path(__file__).parent / "polyhaven_assets.json"


if __name__ == "__main__":
    print(POLYHAVEN_ASSETS_SNAPSHOT_PATH)
