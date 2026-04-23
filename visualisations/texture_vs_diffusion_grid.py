"""
Generate a grid image showing pairs of random texture renders vs diffusion renders.
Layout: 3 rows (one per category), 2 pairs per row (4 images total per row).
Prompt shown underneath each diffusion image; "Random texture render" under each texture image.
"""

import pathlib
import random
import re

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

matplotlib.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 7,
    }
)

BASE_DIR = pathlib.Path("/fast_storage_2/symlinked_homes/tlips/Documents/Documents/diffusing-synthetic-data/data")
RENDERER = "ControlNetTXTFromDepthRenderer_ccs=1.5"

CATEGORIES = {
    "Shoes": {
        "random_textures": BASE_DIR / "coco/shoes/random_textures/images",
        "diffusion": BASE_DIR / f"coco/shoes/diffusion_renders/02-gemini-prompts/{RENDERER}/images",
    },
    "T-shirts": {
        "random_textures": BASE_DIR / "coco/tshirts/random_textures/images",
        "diffusion": BASE_DIR / f"coco/tshirts/diffusion_renders/02-gemini-prompts/{RENDERER}/images",
    },
    "Mugs": {
        "random_textures": BASE_DIR / "coco/mugs/random_textures/images",
        "diffusion": BASE_DIR / f"coco/mugs/diffusion_renders/02-gemini-prompts/{RENDERER}/images",
    },
}

RANDOM_TEXTURE_PROMPT = "Random texture render"


def extract_prompt(filepath: pathlib.Path) -> str:
    """Extract the prompt from a diffusion render filename (strips trailing _{i})."""
    stem = filepath.stem
    match = re.match(r"^(.*?)_(\d+)$", stem)
    return match.group(1) if match else stem


def find_pairs(category_config: dict) -> list[dict]:
    """Find (random_texture_path, diffusion_path, prompt) triples for a category."""
    random_base = category_config["random_textures"]
    diffusion_base = category_config["diffusion"]

    pairs = []
    for mesh_dir in sorted(random_base.iterdir()):
        if not mesh_dir.is_dir():
            continue
        diff_mesh_dir = diffusion_base / mesh_dir.name
        if not diff_mesh_dir.exists():
            continue

        for scene_dir in sorted(mesh_dir.iterdir()):
            if not scene_dir.is_dir():
                continue

            random_texture_dir = scene_dir / "random_textures"
            if not random_texture_dir.exists():
                continue
            random_images = sorted(random_texture_dir.glob("rgb_*.png"))
            if not random_images:
                continue

            # diffusion images are in scene_dir / RENDERER / *.png
            diff_scene_dir = diff_mesh_dir / scene_dir.name / RENDERER
            if not diff_scene_dir.exists():
                continue
            diff_images = sorted(diff_scene_dir.glob("*.png"))
            if not diff_images:
                continue

            pairs.append(
                {
                    "random_texture": random_images[0],
                    "diffusion": diff_images[0],
                    "prompt": extract_prompt(diff_images[0]),
                }
            )

    return pairs


def escape_latex(text: str) -> str:
    """Escape characters that are special in LaTeX."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for char, repl in replacements:
        text = text.replace(char, repl)
    return text


def wrap_text(text: str, max_chars: int = 45) -> str:
    """Wrap text at word boundaries."""
    words = text.split()
    lines, current = [], []
    count = 0
    for word in words:
        if count + len(word) + (1 if current else 0) > max_chars:
            lines.append(" ".join(current))
            current = [word]
            count = len(word)
        else:
            count += len(word) + (1 if current else 0)
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def main(seed: int = 42):
    random.seed(seed)

    n_pair_cols = 3  # pairs across
    n_pair_rows = 2  # pairs down

    # Sample 2 random pairs per category, then shuffle together
    sampled = []
    for config in CATEGORIES.values():
        pairs = find_pairs(config)
        sampled.extend(random.sample(pairs, 2))
    random.shuffle(sampled)

    # GridSpec layout: [img, img, spacer, img, img, spacer, img, img]
    # Small wspace controls intra-pair gap; spacer columns create inter-pair gap.
    import matplotlib.gridspec as gridspec

    spacer_ratio = 0.35  # width of spacer column relative to image column
    col_ratios = []
    for i in range(n_pair_cols):
        col_ratios += [1, 1]
        if i < n_pair_cols - 1:
            col_ratios.append(spacer_ratio)
    n_gs_cols = len(col_ratios)

    fig = plt.figure(figsize=(20 / 2.54, 9 / 2.54))
    gs = gridspec.GridSpec(
        n_pair_rows,
        n_gs_cols,
        width_ratios=col_ratios,
        wspace=0.01,
        hspace=0.45,
        left=0.01,
        right=0.99,
        top=0.99,
        bottom=0.22,  # reserve space at bottom for captions
    )

    # Map (row, pair_col, side) → gridspec column index
    def gs_col(pair_col, side):  # side 0=left, 1=right
        return pair_col * 3 + side  # each pair block = img, img, spacer (3 cols)

    pair_captions = []
    for i, pair in enumerate(sampled):
        row_idx = i // n_pair_cols
        pair_col = i % n_pair_cols

        ax_rt = fig.add_subplot(gs[row_idx, gs_col(pair_col, 0)])
        ax_diff = fig.add_subplot(gs[row_idx, gs_col(pair_col, 1)])

        ax_rt.imshow(mpimg.imread(str(pair["random_texture"])))
        ax_rt.axis("off")

        ax_diff.imshow(mpimg.imread(str(pair["diffusion"])))
        ax_diff.axis("off")

        pair_captions.append((row_idx, pair_col, pair["prompt"], ax_rt, ax_diff))

    # Captions centered under each pair, wrapped to exactly the pair width
    fig.canvas.draw()  # needed to compute axis positions
    avg_char_width_pts = 4.0  # approximate for 8pt serif
    for row_idx, pair_col, prompt, ax_left, ax_right in pair_captions:
        pos_l = ax_left.get_position()
        pos_r = ax_right.get_position()
        x_center = (pos_l.x0 + pos_r.x1) / 2
        y_bottom = pos_l.y0
        pair_width_pts = (pos_r.x1 - pos_l.x0) * fig.get_figwidth() * 72
        max_chars = int(pair_width_pts / avg_char_width_pts)
        fig.text(
            x_center,
            y_bottom - 0.01,
            wrap_text(prompt, max_chars=max_chars),
            ha="center",
            va="top",
            fontsize=8,
        )

    out_path = pathlib.Path(__file__).parent / f"texture_vs_diffusion_grid_seed{seed}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    for seed in [0, 1, 2, 3]:
        main(seed)
