# EffectAfford

Implementation of **EffectAfford: Effect-Equivalent Representation Learning for Actor-Robust Affordance Grounding from Passive Video**.

## Installation

Requires Python `>=3.11,<3.13`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0, pandas 2.2.3. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

Use [EPIC-KITCHENS](https://epic-kitchens.github.io/), Something-Something interaction clips or a verified BridgeData V2 contact evaluation set. Effect labels describe visible state change, not action verbs. Keep actor/scene groups disjoint before drawing training pairs or target queries. BridgeData can be placed entirely in test for frozen transfer.

Create `data/interactions.csv` with `id,path,split,effect,group,verb,supports,participant_group` and optional `domain`. `effect` is an integer in `0..--classes-1` (default eight). `group` is an actor/scene grouping that cannot cross splits. `verb` is the action label used to identify hard negatives. `supports` is the target object's binary effect-support label. `participant_group` supplies the fixed balanced participant-group probe task; use the same registered group vocabulary across train/test. Each effect needs at least three independent training groups so an anchor has two cross-group positives.

Each NPZ contains uint8 `pre`, `post`, `target` RGB crops, binary `source_object`, `target_object`, `actor_mask`, and binary `region`. All images/masks share a fixed spatial size, such as 128×128. Register source pre/post views outside actor masks, reject unreliable registrations, and crop the active object with 20% context before resizing. Target crops use the corresponding target-object support. `region` is a human mask for evaluation; training may instead use a registered contact/state-change proxy restricted to the object support. Record proxy provenance during export. The interaction middle frames are used for object association and actor masks, and are never concatenated into the effect embedding. Optional `contact` is a verified target-frame contact coordinate in `[row,column]` after cropping/resizing.

For unsupported target/effect pairs, use `supports=0` with an empty region. Keep test regions and contact points outside all model inputs. Target masks are used for nuisance compositing and metric normalization, rather than to directly mask the predicted affordance heatmap.

## Training and inference

```bash
effectafford data/interactions.csv --output outputs/full --seed 0 --classes 8
```

## Evaluation

The command trains, calibrates on validation and evaluates held-out interactions.

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
