"""
Ground-truth verification for point-based Gemini outputs.

Checks whether a model's returned point(s) — e.g. Gemini's raw
[{"point": [369, 457], "label": "B"}] — fall inside the SAM-generated
ground-truth mask (from SAM_mask_generator.py) for the object being pointed
at. No model API calls here; we supply the model's already-returned JSON.

load_mask() and is_point_in_mask() are copied unchanged from
pointarena/model_evaluator.py , without the [DEBUG MASK] print statements.

Coordinate handling: Gemini's raw "point" field is [y, x], normalized to a
0-1000 scale, NOT pixel [x, y] -- this matches model_evaluator.py's
call_gemini(), which applies this same swap+rescale to every Gemini model in
that pipeline (including gemini-robotics-er-1.6-preview) before ever
checking a point against a mask. That conversion is applied here by default;
pass --raw-pixels if you're instead feeding in coordinates that are already
plain pixel [x, y].

Functions below are defined in the order main() actually calls them:
load_mask -> load_response -> gemini_point_to_pixels -> is_point_in_mask -> visualize.
verify_point_on_mask.py usable standalone (python verify_point_on_mask.py 
--image ... --mask ... --response ...)
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

VIS_DIR = Path("verified")


def load_mask(mask_path):
    """Load a binary mask from a PNG file."""
    try:
        mask_img = Image.open(mask_path)
        mask_array = np.array(mask_img)

        # Normalize to binary (True/False) mask
        if len(mask_array.shape) == 2:
            binary_mask = mask_array > 127
        elif len(mask_array.shape) == 3:
            binary_mask = np.any(mask_array > 127, axis=2)
        else:
            raise ValueError(f"Unexpected mask shape: {mask_array.shape}")

        return binary_mask
    except Exception as e:
        print(f"Error loading mask {mask_path}: {e}")
        return None


def load_response(response_arg):
    """
    Accept either a raw JSON string, or a path to a .json file containing it.
    """
    candidate_path = Path(response_arg)
    text = candidate_path.read_text(encoding="utf-8-sig") if candidate_path.exists() else response_arg

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_start = text.find("[")
        json_end = text.rfind("]") + 1
        if json_start == -1 or json_end == 0:
            raise
        return json.loads(text[json_start:json_end])


def gemini_point_to_pixels(raw_point, img_width, img_height):
    """
    Convert Gemini's raw "point" field to pixel [x, y].
    Same conversion as model_evaluator.py's call_gemini(): Gemini returns
    [y, x] normalized to a 0-1000 scale, regardless of what the prompt asks
    for, so we swap the order and rescale by the actual image dimensions.
    """
    y, x = raw_point
    pixel_x = (x / 1000.0) * img_width
    pixel_y = (y / 1000.0) * img_height
    return [pixel_x, pixel_y]


def is_point_in_mask(point, mask, img_width, img_height):
    """Check if a point [x, y] (pixel coordinates) falls inside the mask."""
    if mask is None or point is None:
        return False

    x, y = point
    pixel_x, pixel_y = int(x), int(y)

    # Ensure coordinates are within image bounds
    if pixel_y < 0 or pixel_y >= img_height or pixel_x < 0 or pixel_x >= img_width:
        return False

    return bool(mask[pixel_y, pixel_x])


def visualize(image_path, mask, checked_points, output_path):
    """Overlay the mask (blue) and each checked point (green=on mask, red=off mask)."""
    img_array = np.array(Image.open(image_path).convert("RGB"))

    overlay = img_array.copy()
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([30, 144, 255])).astype(np.uint8)
    vis = Image.fromarray(overlay)

    draw = ImageDraw.Draw(vis)
    for label, (x, y), passed in checked_points:
        color = (0, 255, 0) if passed else (255, 0, 0)
        r = 8
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=3)
        draw.text((x + r + 2, y - r), label, fill=color)

    vis.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Check whether a Gemini model's returned point(s) fall inside a ground-truth mask."
    )
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument(
        "--mask", required=True, help="Path to the ground-truth mask PNG (e.g. from SAM_mask_generator.py)"
    )
    parser.add_argument(
        "--response",
        required=True,
        help='Model output: a raw JSON string (e.g. \'[{"point": [369, 457], "label": "B"}]\') '
        "or a path to a .json file containing it",
    )
    parser.add_argument(
        "--label", help="Only check the point(s) with this label (default: check every point in the response)"
    )
    parser.add_argument(
        "--raw-pixels",
        action="store_true",
        help="Treat --response points as already-converted pixel [x, y], instead of "
        "Gemini's raw [y, x] normalized-0-1000 format (the default)",
    )
    parser.add_argument("--no-vis", action="store_true", help="Skip saving the visualization image")
    args = parser.parse_args()

    image_path = Path(args.image)
    mask_path = Path(args.mask)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return
    if not mask_path.exists():
        print(f"Mask not found: {mask_path}")
        return

    img_width, img_height = Image.open(image_path).size

    mask = load_mask(mask_path)
    if mask is None:
        print("Failed to load mask.")
        return

    try:
        points = load_response(args.response)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Could not parse --response as JSON: {e}")
        return

    if args.label:
        # layer 1: exact match - the common/compliant case (e.g. "D")
        matching = [p for p in points if p.get("label") == args.label]
        if not matching:
            # layer 2: fallback for verbose labels that still identify the right
            # object, e.g. "the smallest peg (D)" instead of just "D". 
            pattern = re.compile(rf'\b{re.escape(args.label)}\b', re.IGNORECASE)
            matching = [p for p in points if pattern.search(p.get("label", ""))]
            if matching:
                print(f"Label '{args.label}' matched via fallback (layer 2) match against: {[p.get('label') for p in matching]!r}")
        points = matching
        if not points:
            print(f"No point with label '{args.label}' found in the response.")
            return

    checked_points = []
    passed_count = 0
    for p in points:
        label = p.get("label", "?")
        raw_point = p["point"]

        if args.raw_pixels:
            x, y = raw_point
        else:
            x, y = gemini_point_to_pixels(raw_point, img_width, img_height)

        passed = is_point_in_mask([x, y], mask, img_width, img_height)
        if passed:
            passed_count += 1
        print(f"[{label}] raw={raw_point} -> pixel=({x:.1f}, {y:.1f}) -> {'ON MASK' if passed else 'OFF MASK'}")
        checked_points.append((label, (x, y), passed))

    if not args.no_vis:
        VIS_DIR.mkdir(parents=True, exist_ok=True)
        # include the label in the filename when --label narrowed the check to one object 
        # without it, keep the generic name in that case
        if args.label:
            vis_filename = f"{image_path.stem}_label_{args.label}_verify.jpg"
        else:
            vis_filename = f"{image_path.stem}_verify.jpg"
        vis_path = VIS_DIR / vis_filename
        visualize(image_path, mask, checked_points, vis_path)
        print(f"Visualization saved to {vis_path}")

    print(f"\nResult: {passed_count}/{len(checked_points)} passed")


if __name__ == "__main__":
    main()
