polyhaven materials are used as baseline



To set up the code to render them:
- Install the polyhaven addon and download all the available assets, see [here](https://github.com/Poly-Haven/polyhavenassets) for instructions. This cannot be done headless and hence requires a workstation with a monitor (or screen forwarding). The suggested locatation for the assets is a directory called `polyhaven` in your `blender-assets` directory.
- run the polyhaven HDRI downloader script `polyhaven/polyhaven_hdri_downloader.py`, which will download all the HDRIs in the desired resolution.
- run the `polyhaven/assets_snapshots.py` file to obtain a json for each asset library. The main purpose is to enable reproducibility, both over time on the same machine and for different users/machines.
