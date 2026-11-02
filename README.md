<p align="center">
  <img src="assets/WeatherCity-head.gif" width="99%" style="max-width: 100%; height: auto;" />
</p>

<p align="center">
  <b>WeatherCity: Urban Scene Reconstruction with Controllable Multi-Weather Transformation</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CVPR-2026-blue" height="26"/>
  <a href="https://arxiv.org/abs/2602.22096"><img src="https://img.shields.io/badge/arXiv-2602.22096-b31b1b" height="26"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" height="26"/></a>
</p>

This repository contains the official implementation of **WeatherCity**, a 4D urban scene reconstruction and controllable weather editing framework for autonomous driving simulation.

**Base v5** is the Base-v4 PBR reconstruction jointly trained from scratch with
the display-RGB residual branch. Run it with `bash run_scene788_base_v5.sh`
(scene 788, 100 frames). It retains the native Base-v4 renderer, losses and geometry schedules, adding Gaussian feature8
and a zero-initialized RGB gain/additive CNN with a 2k-camera warmup, 1k ramp and
pure-PBR auxiliary loss. The default 30,001 camera updates match native step
0..30000; no pretrained model initializes the run. It automatically resumes its
own optimizer/density/RNG state and evaluates current pure PBR and hybrid outputs.
Use `--prepare-only` to preview. [Commands and protocol](docs/reconstruction/base_v5.ko.md).

**D1 dynamic G-buffer supervision** adds instance-balanced material/normal losses
to Base v5, with a full-scene visibility/purity ablation. Run
`bash run_scene788_dynamic_gbuffer.sh` to preview B0 / D1B / D1P;
`--execute` starts matched scratch training and dynamic-region evaluation.
Inference rendering is unchanged. [Design and commands](docs/reconstruction/dynamic_gbuffer_design.ko.md).

## Installation
WeatherCity is built on top of [DriveStudio](https://github.com/ziyc/drivestudio) and follows the same environment setup as DriveStudio. For DriveStudio installation, dataset downloading, masks, SMPL processing, and the original OmniRe pipeline, please refer to the [DriveStudio README](https://github.com/ziyc/drivestudio) and dataset docs such as [Waymo](docs/Waymo.md) and [NuScenes](docs/NuScenes.md).

WeatherCity additionally uses VGG features for content consistency loss. Make sure the VGG 19 weight file `vgg_conv.pth` configured in [configs/weathercity.yaml](configs/weathercity.yaml) exists.

## Data Preparation

First preprocess the dataset with DriveStudio. For example, Waymo scenes should be processed following [docs/Waymo.md](docs/Waymo.md), including images, cameras, LiDAR, masks, instances, and human pose files.

After DriveStudio preprocessing, WeatherCity expects images to be organized by weather under `images/`:

```text
data/waymo/processed/training/<scene_idx>/
├── images/
│   ├── raw/
│   │   ├── 000_0.jpg
│   │   ├── 001_0.jpg
│   │   └── ...
│   ├── rain/
│   │   ├── 000_0.jpg
│   │   ├── 001_0.jpg
│   │   └── ...
│   └── snow/
│       ├── 000_0.jpg
│       ├── 001_0.jpg
│       └── ...
├── dynamic_masks/
├── fine_dynamic_masks/
├── sky_masks/
├── lidar/
├── instances/
└── ...
```
where `raw/` contains the original images and `rain/`, `snow/`, etc. contain the edited weather images by [Qwen-Image](https://github.com/QwenLM/Qwen-Image).

Then make sure the dataset config loads the same edited weather folders:

```yaml
# configs/datasets/waymo/1cams.yaml
data:
  pixel_source:
    load_weather: ["rain", "snow"]
```

`raw` is added automatically by the dataloader, so `load_weather` should only contain edited weather names.

## Training

Set the project path and run WeatherCity training with the weather conditions you want to optimize:

```sh
export PYTHONPATH=$(pwd)
python tools/train_feature.py \
    --config_file configs/weathercity.yaml \
    --output_root output \
    --train_weathers rain snow \
    --project waymo_weathercity \
    --run_name 788_1_cam_000_029 \
    dataset=waymo/1cams \
    data.scene_idx=788 \
    data.start_timestep=0 \
    data.end_timestep=29
```

- `--train_weathers rain snow` must match the edited folders under `images/` and `load_weather` in the dataset config.

Training outputs are saved to:

```text
output/<project>/<run_name>/
```

## Evaluation

Render and evaluate a trained checkpoint:

```sh
export PYTHONPATH=$(pwd)
python tools/eval.py \
    --resume_from <path_to_checkpoint> \
    --eval_weathers rain snow
```
Evaluation renders `raw` plus the listed weather conditions. 

## G-buffer material relations: paired training from scratch

The DiffReg-inspired local regularizer uses existing teacher material similarity
to constrain rendered albedo, roughness and metallic within valid static surfaces.
It adds no history renders or inference filtering. The paired runner trains B0
and R independently from dataset initialization for 30,001 updates each, then runs
the original Base v4 test metrics and shared-flow temporal diagnostics.

```bash
# Preview; add --execute to train and evaluate both arms.
.conda/bin/python -m tools.run_diffreg_experiments
```

The [experiment notes](docs/reconstruction/diffreg_material_design.ko.md) document
the sampled local approximation, fixed settings, costs and result paths.

## Sparse material consistency: paired training from scratch

Experiment 2 adds a MaterialRefGS-inspired train-only consistency loss between
rendered albedo, roughness and metallic in neighboring training views. It checks
depth, normals, static masks and reprojection cycles, comparing at most 4,096
sampled surface points with one detached neighbor every eight steps. The loss
ramps from step 3,000 to 4,000; inference keeps the original Base v4 renderer.

```bash
# Preview only: does not create an experiment or launch training.
.conda/bin/python -m tools.run_material_consistency_experiments
```

The B0/T runner prepares independent 30,001-update training from scratch with
the original Base v4 image evaluation and paired temporal diagnostics.
[Settings and implementation limits](docs/reconstruction/material_consistency_design.ko.md)
describe this sampled adaptation and the commands for a later explicit launch.

## Teacher calibration: paired training from scratch

Experiment 3 applies an IIF-inspired bounded gain and bias to the existing
material teachers on static surfaces. Ten parameters per training image absorb
global per-view teacher bias, with identity and weak temporal parameter priors.
The renderer, normal supervision and dynamic-object supervision remain those of
Base v4. No neighboring view is rendered, and inference adds no calibration pass.

```bash
# Preview only; does not launch training or evaluation.
.conda/bin/python -m tools.run_teacher_calibration_experiments
```

The B0/C runner compares fresh 30,001-update training under the original image
and paired temporal evaluation protocol. The [experiment notes](docs/reconstruction/teacher_calibration_design.ko.md)
explain the bounded, train-only adaptation, its limits and commands for later use.

## Frozen teacher confidence: paired training from scratch

Experiment 4 uses SVGF/NRD-inspired geometry checks and temporal material
variance to weight the existing teachers. At step 3,000, the freshly trained
model renders each training view once at quarter resolution to build a frozen
confidence cache. Later updates reuse the cache, with no additional history
renders or inference filtering. Static weights retain mean one, and dynamic
supervision, RGB losses and normal losses keep their original weights.

```bash
# Preview only; does not start training, cache construction or evaluation.
.conda/bin/python -m tools.run_teacher_confidence_experiments
```

The B0/V runner preserves fresh initialization and the Base v4 evaluation
protocol. [Experiment notes](docs/reconstruction/teacher_confidence_design.ko.md)
describe confidence construction, frozen-geometry limitations and later launch
commands. No pretrained scene checkpoint supplies the confidence geometry.

## RDG-inspired G-buffer fusion: paired training from scratch

Experiment 5 uses one quarter-resolution generated history and a small
attention head with separate normal/material and geometry inputs. A bounded
linear-albedo correction preserves native detail, while current normals,
roughness, metallic, lighting and original material supervision stay intact.
The head and scene train jointly after a 5k warmup, with extra training work
every eight steps. Inference also renders a history view; its cost and quality
must be measured.

```bash
# Preview only; does not start training or evaluation.
.conda/bin/python -m tools.run_rdg_fusion_experiments
```

The B0/D comparison keeps Base v4's initialization and evaluation protocol.
[Experiment notes](docs/reconstruction/RDG_FUSION_EXPERIMENT.md) describe the
RDG-inspired adaptation, geometric guards, settings and launch commands.

## G-buffer temporal attention with the original Base v4 evaluation

The new material adapter uses camera/depth reprojection and native-pixel 3×3
candidates from t−1/t−2/t−4. Its final attention weights blend linear albedo,
which is shaded using the current normal, roughness, metallic, view and light.
Dynamic surfaces and uncertain geometry retain the original material. Training
keeps the original Base v4 image losses and unfused G-buffer supervision, adding
an independent geometric temporal loss and a small material anchor. The full
[design and implementation notes](docs/reconstruction/gbuffer_attention_temporal_design.ko.md)
describe the comparison groups and interpretation limits.

Preview the seven-arm frozen pilot without creating output directories or
starting GPU work:

```sh
cd /home/user/Desktop/3ableGS
.conda/bin/python -m tools.run_gbuffer_attention_experiments \
  --mode pilot --output-dir output/pbr/gbuffer_attention/scene788_pilot_v1
```

Add `--execute` to prepare B0/H0, train each of C0/C1/A0/A1/A2 for 2,000 updates,
and evaluate all seven arms sequentially. The default source is the existing
scene 788 / 100f Base v4 checkpoint. `--variants A2` runs the proposed method and
its B0 comparator. `--seeds 0 1 2` repeats learned heads while sharing the same
frozen scene and fixed controls. Use a new output directory when changing options.

For training from dataset initialization, use a separate queue:

```sh
.conda/bin/python -m tools.run_gbuffer_attention_experiments \
  --mode joint --seeds 0 1 2 \
  --output-dir output/pbr/gbuffer_attention/scene788_joint_v1 --execute
```

Joint training compares B0/C0/A0/A1/A2 for each seed, with H0 applied to that
seed's trained B0. The original inclusive 0..30000 loop, teacher maps, split,
density schedule and resolution schedule remain in use. Attention ramps on at
5k–6k and temporal loss at 6k–7k. Scene initialization and sampling RNG are
isolated from head initialization and context renders. Additional history
rendering increases training cost. C1 is available through `--variants`.

Official image scores run through **the original `tools/eval.py`**: native
1920×1280, raw, AA off, original stride-8 test IDs, affine/response, and original
per-image PSNR/SSIM/LPIPS aggregation. Each test image gets generated context at
its actual previous timestamps; previous sparse test images are not relabeled
as adjacent frames. Evaluation uses isolated checkpoint/config snapshots, so
the source results are preserved. Attention settings and weights reload through
the ordinary trainer and checkpoint path.

The separate temporal evaluator uses shared reference flow and reference masks,
reports full-sequence and train→test residuals and edges, and includes history
rendering in random-access latency. This diagnostic contains training frames;
it is not a contiguous held-out temporal benchmark. `--skip-temporal` omits it.
Standalone `tools/eval_gbuffer_attention.py --streaming-cache` additionally
supports a bounded CPU context cache; cache transfers are part of render timing.

Queues write `manifest.json`, `state.json`, `summary.csv`, `summary.json`, logs,
and per-stage attempts. Repeating the same command verifies completed artifacts
and restarts interrupted stages in a fresh attempt. Source checkpoint/config
and code use content hashes; dataset/teacher files use recorded size/mtime.
Joint restarts are not optimizer-state resumes. The standalone frozen pilot
supports exact optimizer/RNG continuation with `tools/train_gbuffer_attention.py
--resume` and identical options; queue retries intentionally preserve the old
attempt instead. Keep inputs unchanged when resuming standalone pilots.

```sh
MPLCONFIGDIR=/tmp/3ablegs-mpl .conda/bin/python -m unittest discover \
  -s tests -p 'test_gbuffer_attention*.py' -v
```

CPU checks cover physical candidate rejection, gradients, original affine,
unfused teacher material, train-only context, checkpoint reload, evaluation
order/cache equality, optimizer registration, failure/restart and exact pilot
resume. CUDA rendering and actual quality/latency remain to be measured on a
working NVIDIA runtime.

## Temporal material transport (Base v4 experiment)

The raw PBR renderer can reproject previous albedo, roughness and metallic onto
confident static surfaces before shading with the current normal, view and light.
This inference-only experiment uses a frozen checkpoint and no teacher G-buffers
or additional training. Dynamic surfaces, disocclusions and uncertain depth
boundaries reject history.

```sh
cd /home/user/Desktop/3ableGS
MPLCONFIGDIR=/tmp/3ablegs-mpl .conda/bin/python tools/render_temporal_material.py \
  --checkpoint output/pbr/rebuild/scene_788_100f/base_v4_raw/checkpoint_final.pth \
  --output-dir output/pbr/temporal/scene788_material \
  --num-frames 100 \
  --downscale 2 \
  --mode material \
  --save-images
```

Use a new output directory for each run. `comparison.mp4` shows baseline,
material transport and history weight from left to right. PNGs are saved under
`baseline/`, `transport/` and `weight/`; `frames.jsonl` and `summary.json` contain
paired PSNR, history coverage, combined rendering time, reference-flow temporal
residual and edge-preservation metrics. `--mode depth`,
`--mode normal` and `--mode material` enable progressively more correspondence
checks. `--max-history 8` caps the effective EMA sample count, not a strict
eight-frame window. Frame indices are relative to the checkpoint's full sequence.

Both RGB outputs use the same mean affine embedding across train/test frames.
These reconstruction-path diagnostics include training frames and are not an
held-out temporal benchmark or directly comparable to historical per-image
affine PSNR. Temporal evaluation uses shared GT RGB optical flow and static masks,
separately from renderer history gates. CUDA is required for scene rendering;
CPU tests can be run with:

```sh
MPLCONFIGDIR=/tmp/3ablegs-mpl .conda/bin/python -m unittest discover \
  -s tests -p 'test_temporal_material*.py' -v
```

### Sequential experiment suite, including teacher regeneration and retraining

The following command executes all 34 stages sequentially on one GPU. It uses
the existing checkpoint for the first comparisons and initializes each subsequent
training run from scratch. The default is one seed and 30,000 iterations per
training run (four runs).

```sh
cd /home/user/Desktop/3ableGS
MPLCONFIGDIR=/tmp/3ablegs-mpl .conda/bin/python tools/run_temporal_experiments.py \
  --checkpoint output/pbr/rebuild/scene_788_100f/base_v4_raw/checkpoint_final.pth \
  --output-dir output/pbr/temporal/suite_scene788_v1 \
  --num-frames 100 \
  --downscale 2 \
  --iterations 30000 \
  --seeds 0 \
  --holdout-ranges 20:30,60:70 \
  --execute
```

| Stages | Comparison |
| --- | --- |
| 15 frozen renders | No history; RGB/depth/normal/material transport; history weights 0.85/0.5/0.3/0.1; nonrecursive history; neighborhood clamp; stricter geometric gates; shuffled/constant normal and roughness guides |
| 6 sparse edits | Geometry/RGB/G-buffer affinity, each using 1 or 3 train keyframes |
| 1 video teacher generation | Ordered video, separate train/test clips, four G-buffer channels |
| 4 training runs | Static teacher, static teacher + temporal loss, video teacher, video teacher + temporal loss |
| 8 holdout renders | Each newly trained model evaluated with no history and material transport with clamp, weight 0.3 |

Remove `--execute` to validate local assets and print the entire plan without
creating files or launching GPU jobs. `--only frozen` runs the first 21 stages.
`--seeds 0,1,2` repeats the four training configurations and their evaluations for
three seeds; frozen comparisons and teacher generation remain shared.
`--save-images` additionally retains per-frame lossless RGB and weight images.
`--downscale` affects evaluation rendering; retraining keeps the saved training
resolution. `--num-frames` limits frozen renders only; teacher generation and
retraining preserve the checkpoint's complete timeline.

After interruption or failure, rerun the **same command**. Completed stages are
verified and skipped. Failed renders and training start in a new attempt
directory, preserving earlier outputs. Training restarts from scratch because
the saved model checkpoint does not contain the optimizer state. Teacher
generation reuses validated complete clips. A changed code/config/input identity
requires a new suite output directory.

The output directory contains `suite.json` (plan, provenance, status and attempts),
`suite_summary.csv` and `suite_summary.json` (paired results), `logs/`,
`renders/<experiment>/attempt_001/comparison.mp4`, and `train/` checkpoints.
Generated teacher maps are in `video_teacher/data/gbuffer_frames/raw`; exact
conditioning frame IDs are recorded in `video_teacher/work/generation.json`.
Default Cosmos paths point to the neighboring
`cosmos-transfer1-diffusion-renderer` checkout and the `cosmos-dr` Conda environment;
override them with `--cosmos-root`, `--cosmos-env`, and `--conda`.

The new holdout is frames **20–29 and 60–69**, with exclusive upper bounds in the
CLI. Train and test frames never share a teacher-generation clip. Short clips
repeat their final frame to reach 57 inputs and discard the padded outputs.
Holdout RGB is excluded from training image selection and Gaussian color
initialization; dynamic pose interpolation uses actual train-valid endpoints.
Full-sequence LiDAR, calibration and object annotations remain available as
side information. Evaluation history starts empty in each heldout block.

The direct mechanisms operate before PBR shading: material history is reprojected
and gated by the current geometry/G-buffer, while sparse edits propagate over a
fixed world-space anchor graph. Guide ablations change correspondence decisions
without changing the material payload or shading normals. The sparse edit is a
controlled albedo/roughness change in a normalized seed ROI, propagated from
train keyframes; it is not a learned weather transfer model. Its raw-reference
PSNR is therefore omitted from the suite comparison CSV.

Reference temporal residual measures
`|(prediction_t - warp(prediction_previous)) - (GT_t - warp(GT_previous))|`,
using shared GT RGB DIS flow with forward/backward consistency and GT static
masks. Edge gradient error and edge strength ratio help detect blur; flow remains
an estimate. The same flow cache is reused across variants. The temporal-loss
training baseline detaches correspondence and the previous target; gradients
through the current material render can still affect Gaussian geometry and
opacity. No heldout score automatically selects a variant.

CPU unit and integration checks (including simulated subprocess failure/restart):

```sh
MPLCONFIGDIR=/tmp/3ablegs-mpl .conda/bin/python -m unittest discover \
  -s tests -p 'test_temporal*.py' -v
MPLCONFIGDIR=/tmp/3ablegs-mpl .conda/bin/python -m unittest \
  tests.test_surface_material_edit tests.test_video_gbuffers -v
```

## Ego-motion temporal material experiments

The sequential runner implements a frozen-base pilot in this order:

1. Depth/pose motion diagnostics and inference-time antialiasing comparison.
2. Five matched-budget material fine-tunes: continued fitting, motion-balanced
   fitting, legacy adjacent-frame A/R/M consistency, robust linear-albedo
   consistency with train-only t−1/2/4 pairs, and optional roughness consistency.
3. CNN comparisons: current frame only, RGB/geometry routing, full G-buffer
   routing, no residual offset, and full G-buffer without temporal loss.
4. Routing-only constant/shuffled normal and roughness ablations, preserving
   transported material payloads.
5. Two-block local Transformer with the same encoder, output heads and history.

G-buffer depth/camera/visibility directly select correspondences. Learned bounded
offsets refine history sampling, gates select valid history, and bounded albedo
residuals are re-shaded using current-view normal/roughness/metallic/light. The
Transformer searches 3×3 neighborhoods in quarter-resolution reprojected feature
maps; it does not use global attention. Roughness prediction and keyframe memory
are not part of this first implementation. Invalid history is hard-masked, and
learned gates never mask the training loss. Both RGB/geometry and full-guide CNNs
have equal parameter counts; geometry-only routing receives zero material slots.

Geometry, camera, normals, lighting and camera response remain frozen. When a
normal decoder shares Gaussian features, only the raw material decoder is
updated so the freeze survives ordinary checkpoint save/reload. Temporal-loss
variants run without inference history. Neural heads use generated inputs only;
ground-truth RGB is supervision and never enters their features.

The default checkpoint is the existing scene 788 contiguous-holdout
`static_teacher_seed0/attempt_001/checkpoint_final.pth`. Its first held-out block
20:30 is validation, and 60:70 is test, with history reset at every boundary.
The original stride-split `base_v4_raw` checkpoint is rejected for this experiment.
Since these scene 788 blocks have already been inspected, their new results are
exploratory. Repeat with additional held-out high/low-motion scenes before making
a generalization claim.

Preview all 36 stages without creating outputs:

```sh
cd /home/user/Desktop/3ableGS
.conda/bin/python -m tools.run_ego_motion_experiments
```

Run all stages sequentially, after the existing rain/snow queue ends, in a
detached process that survives VS Code or terminal closure:

```sh
cd /home/user/Desktop/3ableGS
.conda/bin/python -m tools.run_ego_motion_experiments \
  --output-dir output/pbr/ego_motion/scene788_v1 \
  --wait-for-queue output/reconstruction/scene_788_rain_snow_100f_queue/status.json \
  --execute --detach
```

The default pilot uses seed 0 and 2,000 optimizer steps per training variant
(five material runs and six neural-head runs). It executes every declared
variant, including the Transformer; validation selection does not silently skip
experiments. Remove `--wait-for-queue` to start immediately. Another scene can be
run with `--checkpoint PATH` and a new output directory, provided its saved
configuration already has at least two separated contiguous held-out blocks.
Use `--seeds 0 1 2` in a new directory for repeated seeds. `--groups diagnosis`,
`--groups material`, or `--groups cnn transformer` limits the comparison group;
the shared baseline and diagnostics are always included.

Re-run the same command after interruption: completed artifacts are verified
and skipped; material/head optimizer and RNG state resume from periodic atomic
checkpoints. Code, options or source changes require a new output directory.
Concurrent workers for the same directory are rejected. The existing rain/snow
queue is never stopped; either completed or failed terminal status releases the
wait, while an unfinished queue with a missing worker reports an error.

```sh
tail -f output/pbr/ego_motion/scene788_v1/runner.log
cat output/pbr/ego_motion/scene788_v1/state.json
```

Results include `summary.csv`, validation-only `selection.json`, secondary
`matched_coverage.json`, and per-variant `summary.json`, `frames.jsonl`, predictions
and `comparison.mp4` (base | prediction | accepted history support). Metrics
include PSNR/LPIPS, independent reference-flow temporal residual, edges,
high-motion strata, frozen-base surface regions, coverage and timings.
`unsupported_proxy` is geometric reprojection failure, not ground-truth
disocclusion. Shared-support PSNR is a secondary method-dependent subset;
primary temporal metrics never use a learned acceptance mask. Frame-weighted
latency excludes cache I/O and should be measured without another GPU workload
for deployment comparisons. Selection requires validation temporal improvement
over 1%, PSNR loss no more than 0.05 dB, edge L1 at most 1.02× baseline and edge
strength at least 0.98× baseline; test metrics never determine selection.

```sh
MPLCONFIGDIR=/tmp/3ablegs-mpl .conda/bin/python -m unittest discover \
  -s tests -p 'test_ego*.py' -v
```

## Layered G-buffer surface-mixing experiment

The primary comparison is now **full training from data initialization**, with
the layered renderer active throughout optimization. Frozen renderer swaps and
decoder-only adaptation below are diagnostics, not the final module result.
The paired run trains Base v4 and Base v4 + instance-separated shading with the
same seed, training frames, teachers, original losses, learning-rate schedules,
and Gaussian refinement schedule. Geometry, opacity, material features/decoders,
normals, object motion, lighting and response follow their original optimizer
settings. Camera pose retains its original zero learning rate. The renderer
itself has no extra learned weights; its differentiable outputs train the whole
scene model. AA remains off.

Full training uses `packed=false` and includes every layer's screen-space
gradient in Gaussian refinement. It retains raster graphs and checkpoints
shading to preserve gsplat's AbsGrad signal. Summing per-pass AbsGrad is a
conservative statistic because cancellation between passes is not recovered;
it is not algebraically identical to a fused single-pass AbsGrad kernel.

```sh
MPLCONFIGDIR=/tmp/3ablegs-mpl .conda/bin/python -m tools.run_layered_joint_training \
  --source-config output/pbr/temporal/suite_scene788_v1/train/static_teacher_seed0/attempt_001/config.yaml \
  --output-dir output/pbr/layered_gbuffer/scene788_joint_v1 \
  --steps 30000 --seed 0 --order layered-first --execute
```

Omit `--execute` to inspect the plan. Each arm initializes from the dataset;
neither loads the source checkpoint's weights. Training is followed by evaluation
using **that model's saved renderer**, and separate validation/test aggregation.
The saved configuration restores the renderer for normal checkpoint loading.
The original trainer counts steps from 0 through 30000 inclusive (30001 updates).
Use `--resume` with the same options to continue an interrupted queue: completed,
verified stages are reused, but unfinished training restarts in a new attempt
because the original trainer does not save the optimizer/RNG state. It is not
an exact training-state resume. The 788 scene remains exploratory.

The reference renderer preserves every Gaussian's global visibility weight.
It accumulates material features and a unit indicator per group with **colors
masked, never opacity**, shades each group, and sums the already premultiplied
radiance. Sky, camera response and affine are applied once. It adds no temporal
history or loss. The default four conditions are B0 (original), B1 (single-layer
linear albedo), L0 (grouped sRGB albedo), and L1 (grouped linear albedo).

Run the frozen comparison at the saved native resolution with AA off:

```sh
MPLCONFIGDIR=/tmp/3ablegs-mpl .conda/bin/python -m tools.layered_gbuffer_experiment \
  --checkpoint output/pbr/temporal/suite_scene788_v1/train/static_teacher_seed0/attempt_001/checkpoint_final.pth \
  --output output/pbr/layered_gbuffer/scene788_frozen_v1 \
  --split heldout --grouping instance --save-every 1
```

Add `--layer-budgets 0,2,4 --ablations normal,brdf,albedo` for layer-budget and
component controls. Zero preserves all groups; capped variants keep K-1 dominant
groups plus an overflow group that retains every remaining contribution. Use
`--grouping dynamic` for background versus the union of dynamic nodes. Instance
grouping does not split surfaces inside an object or the background. Binary
reference dynamic masks evaluate union-object boundaries, not touching actors'
individual boundaries. G3 surface-patch construction is a subsequent experiment.

`summary.json` contains pooled PSNR, reference-corrected temporal error, dynamic
and 3/5/9-pixel boundary metrics, full-image SSIM/LPIPS, and conservation checks.
`frames/` contains per-frame records; `images/` contains RGB comparisons, mixing
maps and fixed-transform diffuse/specular previews. All variants use the same
shared mean affine. These scores are not an exact replay of image-specific affine
evaluation. Flow-rejected error is a reconstruction proxy, not GT disocclusion.
This multi-pass renderer is a correctness reference, not an optimized speed result.

For the original stride-split base-v4 checkpoint, explicitly label a diagnostic
run with `--diagnostic-all --split all`. Do not re-label its training observations
as clean held-out frames. Existing output directories are refused; `--restart`
requires matching checkpoint/config/code/options and reconstructs flow history
by rendering the full requested sequence again.

Matched adaptation uses the saved training split, original loss weights and
G-buffer teachers, and updates **raw material decoder weights only**. Features,
geometry, motion, opacity, normals, lighting and camera response remain frozen.
All four variants start from the same checkpoint and use the same frame schedule:

```sh
for variant in B0 B1 L0 L1; do
  MPLCONFIGDIR=/tmp/3ablegs-mpl .conda/bin/python -m tools.finetune_layered_gbuffer \
    --checkpoint output/pbr/temporal/suite_scene788_v1/train/static_teacher_seed0/attempt_001/checkpoint_final.pth \
    --output-dir "output/pbr/layered_gbuffer/adapt_${variant}_s0" \
    --variant "$variant" --grouping instance --steps 3000 --seed 0 || break
done
```

Interrupted adaptation can resume with identical flags plus `--resume`.
Each output includes `checkpoint_final.pth`, original `config.yaml`, and
`layered_config.json` recording the trained renderer. To evaluate an adapted
checkpoint, use the frozen comparison command with its path: it deliberately
evaluates all four renderers, so match the trained condition when comparing the
four adapted models. Cross-renderer entries diagnose renderer compensation.

```sh
MPLCONFIGDIR=/tmp/3ablegs-mpl .conda/bin/python -m unittest discover \
  -s tests -p 'test_layered*.py' -v
```

## Dynamic Weather Effects

After training the multi-weather scene, use the dynamic weather renderer to add controllable rain, snow, or fog:

```sh
export PYTHONPATH=$(pwd)
python tools/add_weather_effects.py \
    --config_file configs/weathercity.yaml \
    --effects_config_file configs/weather_effects.yaml \
    --output_root output_effects \
    --project waymo_weathercity_effects \
    --weathers rain snow fog \
    --run_name 788_1_cam_000_029_effects \
    --load_from <path_to_checkpoint> \
    dataset=waymo/1cams \
    data.scene_idx=788 \
    data.start_timestep=0 \
    data.end_timestep=29
```

Weather effect parameters are controlled in [configs/weather_effects.yaml](configs/weather_effects.yaml).

Rendered videos and frames are saved to:

```text
output_effects/<project>/<run_name>/
```

## Adding New Weather

To add a new weather condition, follow these steps:

1. Generate weather-edited images with the same filenames as the raw images.
2. Put them under the processed scene:

```text
/data/waymo/processed/training/<scene_id>/images/<new_weather>/
├── 000_0.jpg
├── 001_0.jpg
└── ...
```

3. Add the new weather name to the dataset config:

```yaml
# configs/datasets/waymo/1cams.yaml
data:
  pixel_source:
    load_weather: ["rain", "snow", "<new_weather>"]
```

4. Add an optimizer entry for the new weather decoder in [configs/weathercity.yaml](configs/weathercity.yaml):

```yaml
trainer:
  gaussian_optim_general_cfg:
    mlp.<new_weather>:
      lr: 0.0001
```

5. Include the new weather at training and evaluation time.

## Acknowledgments

WeatherCity is based on [DriveStudio](https://github.com/ziyc/drivestudio) and we also use [gsplat](https://github.com/nerfstudio-project/gsplat) for Gaussian rasterization and text-guided image editing models ([Qwen-Image](https://github.com/QwenLM/Qwen-Image)) for weather background generation. The evaluation metrics are referenced from [WeatherEdit](https://github.com/Jumponthemoon/WeatherEdit).

Please also cite DriveStudio/OmniRe if you use this codebase.

## Citation

```bibtex
@misc{wu2026weathercityurbanscenereconstruction,
      title={WeatherCity: Urban Scene Reconstruction with Controllable Multi-Weather Transformation}, 
      author={Wenhua Wu and Huai Guan and Zhe Liu and Hesheng Wang},
      year={2026},
      eprint={2602.22096},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.22096}, 
}
```
