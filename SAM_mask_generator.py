"""
Interactive SAM mask generator.

By default, loops over every image in the local images/ folder and lets you
left-click foreground points (the object to include) and right-click
background points (areas to exclude) on the object you want masked. SAM
returns 3 candidate masks at different granularities; you pick the one that
actually matches, or redo the points, before it's saved to masks/.

Use --image to instead process a single image that lives outside the
images/ folder.

Use --image together with --label to mask one specific labeled object
instead of the whole image (e.g. one peg on a pegboard), saving to
masks/<image_stem>_label_<label>_mask.png. Which label to pass is whatever
distance_questions_generator.py's ground_truth already says is the answer
for that image -- this tool doesn't guess it, you tell it.
"""
import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from segment_utils import SegmentAnythingHelper

IMAGES_DIR = Path("images")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_OUTPUT_DIR = Path("masks")
# Default checkpoint location. Set the SAM_CHECKPOINT_PATH environment variable
# (e.g. in a local .env file, which is gitignored) to point at wherever your
# SAM checkpoint actually lives -- e.g. a sibling pointarena repo, so this
# project doesn't need its own copy of the ~2.5GB file. Falls back to a
# checkpoint in this repo's own root if the env var isn't set. Override either
# way with --checkpoint.
DEFAULT_CHECKPOINT_PATH = os.environ.get("SAM_CHECKPOINT_PATH", "sam_vit_h_4b8939.pth")


def _numeric_sort_key(path):
    """Sort by the number embedded in the filename (img_2 before img_10),
    not by plain string order (which would put img_10 right after img_1)."""
    match = re.search(r"(\d+)", path.stem)
    return (int(match.group(1)), path.stem) if match else (float("inf"), path.stem)


def get_points_from_user(image_path, prompt_text):
    img = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(
        f"{prompt_text}\n"
        "Left-click = include (green), Right-click = exclude (red). Press Enter when done."
    )

    fg_points = []
    bg_points = []

    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        if event.button == 1:
            fg_points.append([x, y])
            ax.plot(x, y, "g+", markersize=12, markeredgewidth=2)
        elif event.button == 3:
            bg_points.append([x, y])
            ax.plot(x, y, "rx", markersize=12, markeredgewidth=2)
        else:
            return
        fig.canvas.draw()

    def on_key(event):
        if event.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    return fg_points, bg_points


def create_overlay(image, mask, alpha=0.5, color=(30, 144, 255)):
    mask_image = np.zeros_like(image, dtype=np.uint8)
    mask_image[mask] = color
    overlay = cv2.addWeighted(image, 1, mask_image, alpha, 0)
    return Image.fromarray(overlay)


def show_candidates(image_path, masks, scores):
    img = np.array(Image.open(image_path))
    fig, axes = plt.subplots(1, len(masks), figsize=(5 * len(masks), 5))
    for i, (mask, score) in enumerate(zip(masks, scores)):
        overlay = create_overlay(img, mask)
        axes[i].imshow(overlay)
        axes[i].set_title(f"Option {i + 1}\nscore: {score:.3f}")
        axes[i].axis("off")
    plt.show()
    plt.close(fig)


def process_image(sam_helper, image_path, mask_path, prompt):
    """Run the interactive click -> SAM predict -> pick-a-mask loop for one image."""
    sam_helper.set_image(str(image_path))

    print(f"\n=== {image_path.name} ===")
    print(f"Prompt: {prompt}")

    while True:
        fg_points, bg_points = get_points_from_user(str(image_path), prompt)
        if not fg_points:
            print("No foreground points clicked, skipping this image.")
            break

        points = fg_points + bg_points
        labels = [1] * len(fg_points) + [0] * len(bg_points)

        masks, scores, _ = sam_helper.predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array(labels, dtype=np.int64),
            multimask_output=True,
        )

        print("Candidate scores:", [f"{s:.3f}" for s in scores])
        show_candidates(str(image_path), masks, scores)

        choice = input("Pick mask option (1/2/3), 'n' to redo points, 's' to skip: ").strip().lower()
        if choice in ("1", "2", "3"):
            idx = int(choice) - 1
            mask_img = sam_helper.create_binary_mask(masks[idx])
            mask_img.save(mask_path)
            print(f"Saved mask to {mask_path}")
            break
        elif choice == "s":
            print("Skipped.")
            break
        # anything else ("n" or otherwise) loops back to re-click points


def main():
    parser = argparse.ArgumentParser(
        description="Interactive SAM mask creation. Defaults to processing every "
        "image in the local images/ folder; use --image for a one-off file "
        "that lives outside that folder."
    )
    parser.add_argument(
        "--image",
        help="Path to a single input image that is NOT inside the local images/ "
        "folder. If omitted, every image inside images/ is processed instead.",
    )
    parser.add_argument(
        "--output",
        help="Path to save the mask PNG. Only valid together with --image "
        "(default: masks/<image_stem>_mask.png). Ignored in folder mode, "
        "where each image always gets its own masks/<image_stem>_mask.png.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Text describing the object to segment, shown as the click-window title. "
        "Default: \"Click the object you want masked.\" -- or, when --label is used, "
        "\"Click the object labeled '<label>'.\"",
    )
    parser.add_argument(
        "--label",
        help="Mask one specific labeled object within --image (e.g. one peg on a "
        "pegboard), instead of the whole image. Requires --image. Saves to "
        "masks/<image_stem>_label_<label>_mask.png.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the SAM checkpoint (.pth) file (default: pointarena repo's checkpoint)",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Also process images that already have a saved mask (overwrites it)",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"SAM checkpoint not found: {checkpoint_path}")
        return
    os.environ["SAM_CHECKPOINT_PATH"] = str(checkpoint_path)

    if args.label and not args.image:
        print("--label requires --image (it masks one labeled object within a specific image).")
        return

    # Build the list of (image_path, mask_path, prompt) jobs to run.
    if args.image:
        # Single-file override, for an image that isn't inside images/.
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            return
        if args.label:
            # One specific labeled object within this image (e.g. one peg).
            mask_path = (
                Path(args.output)
                if args.output
                else DEFAULT_OUTPUT_DIR / f"{image_path.stem}_label_{args.label}_mask.png"
            )
            prompt = args.prompt or f"Click the object labeled '{args.label}'."
        else:
            mask_path = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR / f"{image_path.stem}_mask.png"
            prompt = args.prompt or "Click the object you want masked."
        jobs = [(image_path, mask_path, prompt)]
    else:
        # Default: one job per whole image, looping over every image in images/.
        if not IMAGES_DIR.exists():
            print(f"No '{IMAGES_DIR}' folder found at {IMAGES_DIR.resolve()}")
            return
        found = sorted(
            (p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS),
            key=_numeric_sort_key,
        )
        if not found:
            print(f"'{IMAGES_DIR}' folder is empty. Add images there and rerun.")
            return
        prompt = args.prompt or "Click the object you want masked."
        jobs = [(p, DEFAULT_OUTPUT_DIR / f"{p.stem}_mask.png", prompt) for p in found]

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sam_helper = SegmentAnythingHelper()

    for image_path, mask_path, prompt in jobs:
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if mask_path.exists() and not args.redo:
            print(f"Skipping {image_path.name}: mask already exists at {mask_path}")
            continue
        process_image(sam_helper, image_path, mask_path, prompt)


if __name__ == "__main__":
    main()
