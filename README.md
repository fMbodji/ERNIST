# ERNIST: A CAD-Grounded Benchmark for Multi-Category Embodied Reasoning in Gemini Robotics-ER 1.6

**Status: work in progress.** Built during my third NIST summer internship (Engineering
Laboratory, Intelligent Systems Division, Cognition and Collaboration Systems Group). Not a
finished, peer-reviewed benchmark release; numbers here are preliminary and will be evolving as more test data is collected.

A benchmark testing Gemini Robotics-ER 1.6 across five embodied reasoning categories (Spatial Reasoning, Action Reasoning, Trajectory Reasoning, Pointing, State Estimation), grounded in physical, CAD-designed, 3D-printed peg-in-hole objects. Because ground-truth dimensions and
inter-peg spacing come directly from the CAD design specification used to fabricate the
physical parts (rather than being measured after the fact or estimated), ground truth is exact
to FDM print tolerance. Based on third-party testing of this printer class (Bambu Lab X1 Carbon), well-calibrated dimensional accuracy is typically reported in the ±0.1–0.2mm range; we treat this as an expected upper bound rather than a value measured on our own printed parts. 
Pointing-task scoring extends the [Point-Arena](https://github.com/pointarena/pointarena) evaluation methodology (coordinate normalization convention, mask-based on/off scoring).

## Why This Benchmark

Most spatial-reasoning benchmarks for vision-language models test qualitative relations such as,
("is A left of B") against approximate ground truth: crowd-drawn bounding boxes, model-estimated depth maps, or CAD scenes that were not designed to be manufacturable.
That's usually fine for general visual common sense, but it doesn't answer the question that matters for deploying a VLM as a robot's perception layer: can it produce metric, tolerance-relevant estimates precise enough to drive an actual manipulation action? 
Real assembly tasks like peg-in-hole insertion or pick-and-place run on millimeter-to-centimeter tolerances, not "near" vs. "far." 
ERNIST's ground truth comes directly from the CAD specification used to fabricate the physical pegs and board, rather than being measured or estimated after the fact, so it's exact up to fabrication tolerance instead of approximate.

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
  - [Evaluation Pipeline](#evaluation-pipeline)
  - [Example Run](#example-run)
- [Project Structure](#project-structure)
- [Task Categories](#task-categories)
- [Data Format](#data-format)
- [Current Scope / Known Limitations](#current-scope--known-limitations)
- [Attribution](#attribution)
- [Requirements](#requirements)
- [License](#license)

## Key Features

- **CAD-exact ground truth**: pegs and pegboards designed in Fusion 360 with fully specified
  dimensions, then 3D printed; distances are computed from the CAD spec rather than measured
  post-hoc with calipers or estimated from a photo.
- **Graded clue levels**: distance questions are generated at multiple levels of scale
  information (0 = no clue, 1 = partial cue such as board hole spacing, 2 = full dimensional
  spec), so grounding-dependency can be isolated from raw spatial reasoning.
- **Multi-view test setups**: the same physical board is photographed from multiple camera
  viewpoints (side, mid, overhead, far) to test viewpoint robustness.
- **Automated question generation**: `distance_questions_generator.py` produces distance and
  pointing questions programmatically from `Board`/`Peg` dataclasses, with automatic
  ground-truth computation and cm/mm conversion, instead of hand-writing each question.
- **SAM-grounded pointing evaluation**: pointing ground truth is a human-verified SAM
  segmentation mask per labeled object; a model's predicted point is scored correct if it
  lands inside the correct object's mask.
- **Automated scoring pipeline**: `evaluate_test_results.py` scores an entire results CSV in
  one pass, computing residual error for distance questions and mask hit/miss for pointing
  questions, and prints summary statistics.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/fMbodji/ERNIST.git
cd ERNIST
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the SAM model checkpoint (not included in this repo, ~2.5GB):

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

4. Point `SAM_CHECKPOINT_PATH` at the checkpoint, either as an environment variable or via
   `--checkpoint` when running `SAM_mask_generator.py` (see `segment_utils.py`).

## Usage

### Evaluation Pipeline

The pipeline works end to end like this:

1. **Generate questions.** `distance_questions_generator.py` picks question types and produces a list of questions with computed ground truths 
2. **Query the model.** Each question is tested against Gemini ER 1.6 (currently via Google
AI Studio).
3. **Log results.** Responses are saved into the a CSV file (like the one under data/         Summary_Distance_Questions) which columns such as question type, camera view, and ground truth,   with a `Residual Error` column (difference between ground truth and Gemini's response) and a `Point-on-Mask?` column, both initially empty.
4. **Score.** `evaluate_test_results.py` loads the CSV into a dataframe, call `find_missing_masks` to check whether every pointing question's ground-truth mask already
exists (and prints the exact `SAM_mask_generator.py` command to create any that are
missing), then calls either `get_residual_error` or `check_point_against_mask` per row
depending on the question type, fills in the two scoring columns, saves the scored CSV,
and prints summary statistics (mean/median error, 90th percentile for distance questions;
number and percentage on-mask for pointing questions).

If any pointing-question masks are missing, the script stops before scoring so you don't
score against incomplete ground truth. Run the printed `SAM_mask_generator.py` commands, then
re-run the evaluation script.

### Example Run

A full run, starting from a CSV with six missing pointing masks:

```
$ python evaluate_test_results.py --csv data/Summary_Distance_Questions.csv --output data/Summary_Distance_Questions_scored.csv

6 missing mask(s):
  Missing: masks/setup2_side_view_label_A_mask.png
    Run: python SAM_mask_generator.py --image images/setup2_side_view.jpg --label A
  Missing: masks/setup2_mid_view_label_A_mask.png
    Run: python SAM_mask_generator.py --image images/setup2_mid_view.jpg --label A
  Missing: masks/setup2_ovhd_view_label_A_mask.png
    Run: python SAM_mask_generator.py --image images/setup2_ovhd_view.jpg --label A
  Missing: masks/setup2_side_view_label_E_mask.png
    Run: python SAM_mask_generator.py --image images/setup2_side_view.jpg --label E
  Missing: masks/setup2_mid_view_label_E_mask.png
    Run: python SAM_mask_generator.py --image images/setup2_mid_view.jpg --label E
  Missing: masks/setup2_ovhd_view_label_E_mask.png
    Run: python SAM_mask_generator.py --image images/setup2_ovhd_view.jpg --label E

6 mask(s) need to be created before scoring can proceed.
Run the command(s) printed above, then re-run this script.
```

Creating the missing masks (SAM proposes three candidate masks per click; you pick the one
that actually matches, or press `n` to redo the points if none look right):

```
$ python SAM_mask_generator.py --image images/setup2_side_view.jpg --label A
SAM model initialized on cpu

=== setup2_side_view.jpg ===
Prompt: Click the object labeled 'A'.
Candidate scores: ['0.909', '0.996', '0.925']
Pick mask option (1/2/3), 'n' to redo points, 's' to skip: 2
Saved mask to masks\setup2_side_view_label_A_mask.png

$ python SAM_mask_generator.py --image images/setup2_mid_view.jpg --label A
SAM model initialized on cpu

=== setup2_mid_view.jpg ===
Prompt: Click the object labeled 'A'.
Candidate scores: ['0.972', '0.993', '0.902']
Pick mask option (1/2/3), 'n' to redo points, 's' to skip: 2
Saved mask to masks\setup2_mid_view_label_A_mask.png

$ python SAM_mask_generator.py --image images/setup2_ovhd_view.jpg --label A
SAM model initialized on cpu

=== setup2_ovhd_view.jpg ===
Prompt: Click the object labeled 'A'.
Candidate scores: ['0.925', '1.000', '0.924']
Pick mask option (1/2/3), 'n' to redo points, 's' to skip: 2
Saved mask to masks\setup2_ovhd_view_label_A_mask.png

$ python SAM_mask_generator.py --image images/setup2_side_view.jpg --label E
SAM model initialized on cpu

=== setup2_side_view.jpg ===
Prompt: Click the object labeled 'E'.
Candidate scores: ['0.922', '0.992', '0.937']
Pick mask option (1/2/3), 'n' to redo points, 's' to skip: n
Candidate scores: ['0.980', '0.969', '0.920']
Pick mask option (1/2/3), 'n' to redo points, 's' to skip: 2
Saved mask to masks\setup2_side_view_label_E_mask.png

$ python SAM_mask_generator.py --image images/setup2_mid_view.jpg --label E
SAM model initialized on cpu

=== setup2_mid_view.jpg ===
Prompt: Click the object labeled 'E'.
Candidate scores: ['0.946', '0.941', '0.712']
Pick mask option (1/2/3), 'n' to redo points, 's' to skip: 1
Saved mask to masks\setup2_mid_view_label_E_mask.png

$ python SAM_mask_generator.py --image images/setup2_ovhd_view.jpg --label E
SAM model initialized on cpu

=== setup2_ovhd_view.jpg ===
Prompt: Click the object labeled 'E'.
Candidate scores: ['0.844', '0.660', '0.637']
Pick mask option (1/2/3), 'n' to redo points, 's' to skip: 1
Saved mask to masks\setup2_ovhd_view_label_E_mask.png
```

Re-running the evaluation script now that every mask exists:

```
$ python evaluate_test_results.py --csv data/Summary_Distance_Questions.csv --output data/Summary_Distance_Questions_scored.csv

All referenced masks already exist.
Label mismatch: model returned ['H'], expected 'D'.
Label 'D' matched via fallback (layer 2) match against: ['the smallest peg (D)']
Label 'E' matched via fallback (layer 2) match against: ['the smallest peg (E)']
Label 'E' matched via fallback (layer 2) match against: ["the smallest peg's label (E)"]

Scored CSV saved to data/Summary_Distance_Questions_scored.csv

Distance questions: 103 scored
  Mean absolute residual error: 1.56 cm
  Mean signed residual error: -0.53 cm (positive = model underestimates)
  Median absolute percentage error: 18.6%
  90th percentile absolute error: 3.42 cm
Pointing questions: 12 scored, 66.7% on mask
```

The "Label matched via fallback" lines show the layer-2 label matching: Gemini sometimes returns a verbose label like `"the smallest peg (D)"` instead of the bare `"D"` expected in ground truth. The evaluator falls back to a regex search for the expected label as a whole word inside the returned string before declaring a mismatch, so these still score as correct label matches rather than false negatives.

## Project Structure

All pipeline scripts live at the repo root, and `images/`, `masks/`, and `verified/` are separate top-level folders :

```
distance_questions_generator.py   Generates distance/pointing questions from Board/Peg
                                   dataclasses, with configurable clue levels (0, 1, 2)
                                   and automatic ground-truth computation from known
                                   peg/board dimensions.
SAM_mask_generator.py             Interactive SAM-based mask generation for pointing
                                   ground truth (human-clicked foreground/background
                                   points; supports whole-image or single-labeled-object
                                   masking).
segment_utils.py                  SegmentAnythingHelper wrapper class used by the mask
                                   generator (model loading, mask prediction, overlay
                                   utilities).
evaluate_test_results.py          Scores a results CSV: residual error for distance
                                   questions, mask-based pass/fail for pointing questions.
                                   Prints summary statistics.
verify_point_on_mask.py           Coordinate conversion and point-in-mask checking
                                   (load_mask, load_response, gemini_point_to_pixels,
                                   is_point_in_mask), adapted from Point-Arena's
                                   model_evaluator approach.
requirements.txt                  Pinned/loose dependency list (see Requirements below).

data/
  Summary_Distance_Questions.csv          Generated question set (unscored).
  Summary_Distance_Questions_scored.csv   Scored, with Residual Error / Point-on-Mask
                                           columns filled in.

images/      Raw photographs of the physical pegboard, multiple camera views
             (side, mid, overhead, far), flat — e.g. images/setup1_side_view.jpg.

masks/       SAM-generated binary segmentation masks for individual pegs, flat — e.g.
             masks/setup1_side_view_label_B_mask.png.

verified/    Visual overlays showing Gemini's predicted point against the ground-truth
             mask, for manual sanity-checking (output of verify_point_on_mask.py).
```

## Task Categories

Five reasoning categories are in scope: Distance/Spatial Reasoning, Action Reasoning,
Trajectory Reasoning, Pointing, and State Estimation. Distance and Pointing are the most
developed so far; Action, Trajectory, and State Estimation question generation is still in
progress. See the project report for the full category definitions and example prompts (not
included in this repo pending internal review).

## Data Format

Each row of the results CSV includes:

- `Question`: the exact prompt sent to the model.
- `Type`: question type (e.g., "Distance Given Board Spacing (linear distance)", "Pointing
  Question (largest peg)").
- `Image` / `Camera View`: which setup/view the question was posed against.
- `Ground Truth`: for distance questions, a plain value (e.g., `"13.12 cm"`); for pointing
  questions, a JSON payload with `label`, `mask_path`, and the physical coordinates.
- `Response`: the model's raw output.
- `Residual Error` / `Point on Mask?`: filled in by `evaluate_test_results.py`.

## Current Scope / Known Limitations

- Pilot-scale sample size: distance questions are further along (100+ scored) than the other
  four categories (pointing is at low double digits; Action/Trajectory/State Estimation are
  not yet built out as scored categories).
- Single model evaluated (Gemini Robotics-ER 1.6). No baseline or second-model comparison yet.
- Single annotator for mask ground truth.
- Two physical board setups tested so far.

## Attribution/Citation

This evaluation pipeline adapts the `model_evaluator` approach, SAM checkpoint, and
segmentation utilities from [Point-Arena](https://github.com/pointarena/pointarena)
(Cheng, Duan, et al., *PointArena: Probing Multimodal Grounding Through Language-Guided
Pointing*, [arXiv:2505.09990](https://arxiv.org/abs/2505.09990)), which is also the reference
implementation for the Point-Bench benchmark this project builds on.

## Requirements

Core dependencies:

- PyTorch
- OpenCV, Pillow, Matplotlib for image processing
- Segment Anything Model from Meta AI
- Pandas, NumPy for data handling
- python-dotenv for configuration

See `requirements.txt` for exact package list.

## License

TBD, pending internal review. Do not redistribute without checking current status of this
notice.
