# EffectAfford

Implementation of **EffectAfford: Effect-Equivalent Representation Learning for Actor-Robust Affordance Grounding from Passive Video**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Functionality is checked on Python 3.13.5, PyTorch 2.7.1+cu118 (CPU) with `python smoke_check.py`. The generated-data check uses the explicit CNN control with eight-frame pre/post windows; it exercises NPZ/CSV loading, contrastive/grounding gradients and updates, one training epoch, calibration/evaluation, contact metrics, state reload and frozen-checkpoint evaluation. The pretrained ViT path, real video preprocessing and research performance have not been run here. `send2trash` is an optional check dependency, used to recycle generated inputs.

## Data and preprocessing

Use [EPIC-KITCHENS](https://epic-kitchens.github.io/), Something-Something interaction clips or a verified BridgeData V2 contact evaluation set. Effect labels describe visible state change, not action verbs. Keep actor/scene groups disjoint before drawing training pairs or target queries. BridgeData can be placed entirely in test for frozen transfer.

Create `data/interactions.csv` with `id,path,split,effect,group,verb,supports,participant_group` and optional `domain`. `effect` is an integer in `0..--classes-1` (default eight). `group` is an actor/scene grouping that cannot cross splits. `verb` is the action label used to identify hard negatives. `supports` is the target object's binary effect-support label. `participant_group` supplies the fixed balanced participant-group probe task; use the same registered group vocabulary across train/test. Each effect needs at least three independent training groups so an anchor has two cross-group positives.

Each NPZ contains uint8 `pre`, `post` eight-frame RGB windows (`T,H,W,3`, T=8), a `target` RGB crop (`H,W,3`), binary `source_object`, `target_object`, `actor_mask`, and binary `region`. Images/masks share a fixed spatial size; the manuscript uses 224×224. Single-frame pre/post archives remain supported as an explicit reduced-input control. NPZ paths are relative to the CSV directory. Register source pre/post views outside actor masks, reject unreliable registrations, and crop the active object with 20% context before resizing. Target crops use the corresponding target-object support. `region` is a human mask for evaluation; training may instead use a registered contact/state-change proxy restricted to the object support. Record proxy provenance during export. The interaction middle frames are used for object association and actor masks, and are never concatenated into the effect embedding. Optional `contact` is a verified target-frame contact coordinate in `[row,column]` after cropping/resizing.

For unsupported target/effect pairs, use `supports=0` with an empty region. Keep test regions and contact points outside all model inputs. Target masks are used for nuisance compositing and metric normalization, rather than to directly mask the predicted affordance heatmap.

## Training and inference

```bash
effectafford data/interactions.csv --output outputs/full --seed 0 --classes 8
```

## Evaluation

The default command trains, calibrates on validation and evaluates held-out interactions. `--backbone vit_b_16` is the default and downloads public torchvision ImageNet ViT-B/16 weights on first training use. Temporal features average the observed frame-window patch grids. `--backbone cnn` is a small functional control, not the paper backbone. Saved-checkpoint evaluation restores the architecture, weights, validation threshold, temperature and actor probe without fitting them again:

```bash
effectafford data/interactions.csv --checkpoint outputs/full/model.pt --output outputs/evaluation --seed 0
```

The CSV retains train entries as the fixed retrieval gallery and test entries as queries; test region/contact masks enter metrics only. No model input contains those labels.

## Comparisons and ablations

```bash
effectafford data/interactions.csv --variant appearance_only --output outputs/appearance --seed 0
effectafford data/interactions.csv --variant no_equivalence --output outputs/no_equivalence --seed 0
effectafford data/interactions.csv --variant no_hard --output outputs/no_hard --seed 0
effectafford data/interactions.csv --variant no_crossverb --output outputs/no_crossverb --seed 0
effectafford data/interactions.csv --variant no_consistency --output outputs/no_consistency --seed 0
```

## Outputs

`model.pt` retains fitted weights, threshold and temperature. `history.csv`, `actor_probe.pkl`, `predictions.csv`, `effects.png` and `metrics.json` retain losses, probe, localization/support predictions and metrics. Verified contact fields enable offline contact-proposal metrics.

## Paper-to-code entry points

| Paper module / metric | Code | Command / setting |
| --- | --- | --- |
| Aligned object state difference and eight-frame pooling | `ViTFeatures`, `encode_window`, `embed` | default pretrained ViT-B/16, 20% crop/registration performed before export |
| Effect equivalence / same-verb hard negative | `contrastive` | temperature .07, margin .20; weights .20/.10 |
| Effect-conditioned grounding/support | `ground` | BCE+Dice+.25 boundary; effect/support weight .50 |
| Actor/background consistency with excluded edit boundaries | `intervention`, `js_heatmap` | weight .15; valid support applied to cIoU and altered IoU |
| mIoU/AP/PG/peak distance/retrieval, group probe, worst group, ECE/contact | `main` evaluation | `--minimum-group 25`, `--contact-radius .05` |

The implementation uses mean-pooled window features and a compact 64-channel grounder; its topology is an explicit choice where the manuscript leaves dimensions unspecified. Two AdamW parameter groups use backbone LR 2e-5 and head LR 2e-4, weight decay .05. The current sampler forms three groups per effect per update and trains for the requested epochs; it does not yet implement the manuscript's batch-64 schedule, five-epoch warmup or early stopping on validation mIoU/retrieval. The actor/scene registration/tracking stage and external affordance baselines require their source pipelines and human-verified annotations. These commands do not reconstruct the original experiment evidence, and no paper scores are prefilled. Contact Hit/Proposal-Success additionally require the verified `contact` field.

```bash
python -m pip install send2trash
python smoke_check.py
```
