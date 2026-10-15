# EffectAfford

Implementation of **EffectAfford: Effect-Equivalent Representation Learning for Actor-Robust Affordance Grounding from Passive Video**.

The representation combines aligned pre/post object tokens through signed/absolute changes and contextual features. Cross-group same-effect positives and same-action/different-effect hard negatives shape the embedding. A separate target-image grounder predicts an effect-conditioned region and effect support. Actor removal and background replacement constrain nuisance sensitivity, while validation temperature scaling calibrates support confidence.

## Installation

```bash
python -m venv .venv
python -m pip install -e .
```

Use Python 3.11 or 3.12 and activate the environment in your shell.

## Prepare interactions

Use [EPIC-KITCHENS](https://epic-kitchens.github.io/), Something-Something interaction clips or a verified BridgeData V2 contact evaluation set. Effect labels describe visible state change, not action verbs. Keep actor/scene groups disjoint before drawing training pairs or target queries. BridgeData can be placed entirely in test for frozen transfer.

Create `data/interactions.csv` with `id,path,split,effect,group,verb,supports,participant_group` and optional `domain`. `effect` is an integer in `0..--classes-1` (default eight). `group` is an actor/scene grouping that cannot cross splits. `verb` is the action label used to identify hard negatives. `supports` is the target object's binary effect-support label. `participant_group` supplies the fixed balanced participant-group probe task; use the same registered group vocabulary across train/test. Each effect needs at least three independent training groups so an anchor has two cross-group positives.

Each NPZ contains uint8 `pre`, `post`, `target` RGB crops, binary `source_object`, `target_object`, `actor_mask`, and binary `region`. All images/masks share a fixed spatial size, such as 128×128. Register source pre/post views outside actor masks, reject unreliable registrations, and crop the active object with 20% context before resizing. Target crops use the corresponding target-object support. `region` is a human mask for evaluation; training may instead use a registered contact/state-change proxy restricted to the object support. Record proxy provenance during export. The interaction middle frames are used for object association and actor masks, and are never concatenated into the effect embedding. Optional `contact` is a verified target-frame contact coordinate in `[row,column]` after cropping/resizing.

For unsupported target/effect pairs, use `supports=0` with an empty region. Keep test regions and contact points outside all model inputs. Target masks are used for nuisance compositing and metric normalization, rather than to directly mask the predicted affordance heatmap.

## Train, calibrate and evaluate

```bash
effectafford data/interactions.csv --output outputs/full
```

`--epochs 60`, `--seed`, `--classes 8` and `--minimum-group 25` control training and worst-group eligibility. Every batch samples three independent groups per effect. The source encoder is a shared compact convolutional network; signed/absolute differences and both contexts feed the state-change projector. Object support attention pools the normalized embedding. The target decoder combines effect-conditioned spatial attention with query features. BCE, Dice and boundary losses supervise localization; binary support and effect-class losses provide semantic supervision.

```bash
effectafford data/interactions.csv --variant appearance_only --output outputs/appearance
effectafford data/interactions.csv --variant no_equivalence --output outputs/no_equivalence
effectafford data/interactions.csv --variant no_hard --output outputs/no_hard
effectafford data/interactions.csv --variant no_crossverb --output outputs/no_crossverb
effectafford data/interactions.csv --variant no_consistency --output outputs/no_consistency
```

Appearance-only replaces post features by pre features. `no_crossverb` restricts positive pairs to the same verb. Remaining variants remove their named objective while retaining the same encoders/heads. Nuisance interventions are deterministic: actor pixels are locally inpainted; background outside dilated object support comes from the first independent group in the same split. Prediction consistency excludes a compositing-boundary band and skips interventions overlapping the supervised region.

## Metrics and files

`history.csv` records actual training losses. `model.pt` contains parameters, validation-selected heatmap threshold and confidence temperature. `actor_probe.pkl` stores the frozen-embedding linear probe. `predictions.csv` records original/intervened localization, effect support and optional contact proposals; `effects.png` compares mean IoU by effect.

`metrics.json` reports pixel AP averaged over evaluable target masks, mIoU, pointing-game accuracy, peak-to-region distance normalized by target-object box diagonal, effect macro-F1 and group-disjoint same-effect retrieval mAP. It also reports actor/background cIoU, their mIoU changes, participant-probe balanced accuracy, worst eligible effect×group mIoU and 15-bin support ECE after validation temperature scaling. Probe chance is determined by the registered participant vocabulary; eight balanced groups give 12.5%. A missing eligible group population is `null`.

Verified contact records additionally produce Hit@1/5, Proposal Success@1/5 and normalized expert-contact distance. Proposals are distinct local maxima separated by at least nine pixels. `--contact-radius 0.05` sets the normalized contact radius. Proposal success requires both contact proximity and manipulated-object support. Use `--test-domain bridge` to evaluate only BridgeData test rows while fitting on the same training/validation records. These are offline proposal metrics and do not substitute for executed robot control.

The compact implementation uses supplied alignment/track annotations and compact learned visual encoders rather than downloading a pretrained backbone. Contrastive temperature is 0.1 and hard margin is 0.2. Objective weights follow the method specification: equivalence 0.20, hard negatives 0.10, nuisance consistency 0.15, support/effect supervision 0.50 and boundary 0.25. No test outcome selects a threshold, temperature, pair or probe fit.
