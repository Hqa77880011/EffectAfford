import argparse
import json
import pickle
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score


class EffectAfford(torch.nn.Module):
    def __init__(self, classes, variant):
        super().__init__()
        self.variant = variant
        self.encoder = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 5, stride=2, padding=2), torch.nn.ReLU(), torch.nn.Conv2d(32, 64, 3, stride=2, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU())
        self.change = torch.nn.Sequential(torch.nn.Conv2d(256, 64, 1), torch.nn.GroupNorm(8, 64), torch.nn.ReLU(), torch.nn.Conv2d(64, 64, 1))
        self.query = torch.nn.Linear(64, 64)
        self.key = torch.nn.Conv2d(64, 64, 1)
        self.decode = torch.nn.Sequential(torch.nn.Conv2d(192, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(64, 1, 1))
        self.support = torch.nn.Linear(128, 1)
        self.classify = torch.nn.Linear(64, classes)

    def embed(self, pre, post, support):
        before, after = self.encoder(pre), self.encoder(post)
        if self.variant == "appearance_only":
            after = before
        difference = after - before
        change = self.change(torch.cat([difference, difference.abs(), before, after], 1))
        mask = torch.nn.functional.interpolate(support, size=change.shape[-2:], mode="nearest")
        pooled = (change * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1)
        return torch.nn.functional.normalize(pooled, dim=1)

    def ground(self, z, target):
        feature = self.encoder(target)
        b, d, h, w = feature.shape
        association = torch.einsum("bd,bdhw->bhw", self.query(z), self.key(feature)) / np.sqrt(d)
        attention = torch.softmax(association.flatten(1), dim=1).reshape(b, 1, h, w)
        context = (attention * feature).sum((2, 3))
        joined = torch.cat([feature, z[:, :, None, None].expand(-1, -1, h, w), context[:, :, None, None].expand(-1, -1, h, w)], 1)
        heatmap = torch.nn.functional.interpolate(self.decode(joined), size=target.shape[-2:], mode="bilinear", align_corners=False)
        confidence = self.support(torch.cat([z, feature.mean((2, 3))], 1)).squeeze(1)
        return heatmap, confidence


def load_record(path):
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k] for k in data.files}
    return result


def image_tensor(images):
    return torch.tensor(np.stack(images).transpose(0, 3, 1, 2), dtype=torch.float32) / 255


def batch(records):
    return image_tensor([r["pre"] for r in records]), image_tensor([r["post"] for r in records]), image_tensor([r["target"] for r in records]), torch.tensor(np.stack([r["source_object"] for r in records]), dtype=torch.float32)[:, None], torch.tensor(np.stack([r["region"] for r in records]), dtype=torch.float32)[:, None]


def intervention(record, donor, kind):
    target = record["target"].copy()
    actor = record["actor_mask"].astype(np.uint8)
    support = cv2.dilate(record["target_object"].astype(np.uint8), np.ones((9, 9), np.uint8))
    if kind == "actor":
        target = cv2.inpaint(target, actor, 3, cv2.INPAINT_TELEA)
        edited = actor
    else:
        background = cv2.resize(donor["target"], (target.shape[1], target.shape[0]))
        target[support == 0] = background[support == 0]
        edited = 1 - support
    edge = cv2.morphologyEx(edited, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    valid = 1 - cv2.dilate(edge, np.ones((3, 3), np.uint8))
    if np.any(edited.astype(bool) & record["region"].astype(bool)):
        valid[:] = 0
    return target, valid


def js_heatmap(a, b, valid):
    a = a.sigmoid() * valid
    b = b.sigmoid() * valid
    a = a.flatten(1) / a.flatten(1).sum(1, keepdim=True).clamp_min(1e-8)
    b = b.flatten(1) / b.flatten(1).sum(1, keepdim=True).clamp_min(1e-8)
    middle = (a + b) / 2
    return .5 * ((a * torch.log((a + 1e-8) / (middle + 1e-8))).sum(1) + (b * torch.log((b + 1e-8) / (middle + 1e-8))).sum(1))


def contrastive(z, labels, groups, verbs, variant):
    similarity = z @ z.T
    losses, hard = [], []
    for i in range(len(z)):
        positives = (labels == labels[i]) & (groups != groups[i])
        if variant == "no_crossverb":
            positives &= verbs == verbs[i]
        positive_ids = np.where(positives)[0]
        if not len(positive_ids):
            continue
        denominator = torch.logsumexp(torch.cat([similarity[i, :i], similarity[i, i + 1:]]) / .1, dim=0)
        losses.append(-(similarity[i, positive_ids] / .1 - denominator).mean())
        negative_ids = np.where((verbs == verbs[i]) & (labels != labels[i]))[0]
        if len(negative_ids):
            hard.append(torch.relu(.2 + similarity[i, negative_ids].max() - similarity[i, positive_ids].min()))
    zero = z.sum() * 0
    return torch.stack(losses).mean() if losses else zero, torch.stack(hard).mean() if hard else zero


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--output", default="outputs/full")
    p.add_argument("--classes", type=int, default=8)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--minimum-group", type=int, default=25)
    p.add_argument("--contact-radius", type=float, default=.05)
    p.add_argument("--variant", choices=["full", "appearance_only", "no_equivalence", "no_hard", "no_crossverb", "no_consistency"], default="full")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-domain")
    args = p.parse_args()
    manifest = pd.read_csv(args.manifest)
    if not all((manifest.split == s).any() for s in ("train", "val", "test")):
        raise ValueError("Provide independent train, val and test interactions")
    if manifest.groupby("group").split.nunique().max() != 1:
        raise ValueError("Actor/scene groups must not cross splits")
    records = [dict(load_record(row.path), **row._asdict()) for row in manifest.itertuples(index=False)]
    training = [i for i, r in enumerate(records) if r["split"] == "train"]
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = EffectAfford(args.classes, args.variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    effect_groups = {}
    for i in training:
        effect_groups.setdefault(records[i]["effect"], {}).setdefault(str(records[i]["group"]), []).append(i)
    if any(len(g) < 3 for g in effect_groups.values()):
        raise ValueError("Each training effect needs at least three independent groups for two cross-group positives")
    representatives = {}
    for i, r in enumerate(records):
        representatives.setdefault(r["split"], {}).setdefault(r["group"], i)
    donors = {}
    for i, r in enumerate(records):
        options = [j for group, j in representatives[r["split"]].items() if group != r["group"]]
        if not options:
            raise ValueError("Each split needs at least two groups for background interventions")
        donors[i] = options[0]
    history = []
    for epoch in range(args.epochs):
        epoch_losses = []
        for _ in range(max(1, len(training) // (3 * len(effect_groups)))):
            ids = []
            for groups in effect_groups.values():
                for group in rng.choice(list(groups), 3, replace=False):
                    ids.append(int(rng.choice(groups[group])))
            chosen = [records[i] for i in ids]
            pre, post, target, support, truth = batch(chosen)
            z = model.embed(pre, post, support)
            logits, confidence = model.ground(z, target)
            probability = logits.sigmoid()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, truth)
            loss += 1 - (2 * (probability * truth).sum() + 1) / (probability.sum() + truth.sum() + 1)
            edge_target = torch.tensor(np.stack([cv2.morphologyEx(r["region"].astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) for r in chosen]), dtype=torch.float32)[:, None]
            edge_prediction = torch.nn.functional.max_pool2d(probability, 3, 1, 1) + torch.nn.functional.max_pool2d(-probability, 3, 1, 1)
            loss += .25 * torch.nn.functional.l1_loss(edge_prediction, edge_target)
            labels = np.array([r["effect"] for r in chosen])
            groups = np.array([r["group"] for r in chosen])
            verbs = np.array([r["verb"] for r in chosen])
            eq, hard = contrastive(z, labels, groups, verbs, args.variant)
            if args.variant != "no_equivalence":
                loss += .2 * eq
            if args.variant != "no_hard":
                loss += .1 * hard
            binary = torch.tensor([r["supports"] for r in chosen], dtype=torch.float32)
            loss += .5 * (torch.nn.functional.binary_cross_entropy_with_logits(confidence, binary) + torch.nn.functional.cross_entropy(model.classify(z), torch.tensor(labels, dtype=torch.long)))
            if args.variant != "no_consistency":
                for kind in ("actor", "background"):
                    altered = [intervention(records[i], records[donors[i]], kind) for i in ids]
                    modified, modified_confidence = model.ground(z, image_tensor([r[0] for r in altered]))
                    valid = torch.tensor(np.stack([r[1] for r in altered]), dtype=torch.float32)[:, None]
                    allowed = (valid.sum((1, 2, 3)) > 0).float()
                    loss += .15 * (js_heatmap(logits, modified, valid).mean() + ((confidence.sigmoid() - modified_confidence.sigmoid()).abs() * allowed).sum() / allowed.sum().clamp_min(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch, loss=float(np.mean(epoch_losses))))
    model.eval()
    results = []
    with torch.no_grad():
        for i, r in enumerate(records):
            pre, post, target, support, truth = batch([r])
            z = model.embed(pre, post, support)
            logits, confidence = model.ground(z, target)
            altered = {}
            for kind in ("actor", "background"):
                image, valid = intervention(r, records[donors[i]], kind)
                pred, _ = model.ground(z, image_tensor([image]))
                altered[kind] = pred.sigmoid()[0, 0].numpy() if valid.any() else None
            results.append(dict(index=i, embedding=z[0].numpy(), probability=logits.sigmoid()[0, 0].numpy(), support_logit=float(confidence), effect_prediction=int(model.classify(z).argmax()), **altered))
    val = [r for r in results if records[r["index"]]["split"] == "val"]
    def iou(pred, truth):
        return float(np.sum(pred & truth) / max(1, np.sum(pred | truth)))
    threshold = float(max(np.linspace(.05, .95, 37), key=lambda t: np.mean([iou(r["probability"] > t, records[r["index"]]["region"].astype(bool)) for r in val])))
    def temperature_loss(log_temperature):
        x = torch.tensor([r["support_logit"] for r in val]) / np.exp(log_temperature)
        y = torch.tensor([records[r["index"]]["supports"] for r in val], dtype=torch.float32)
        return float(torch.nn.functional.binary_cross_entropy_with_logits(x, y))
    temperature = float(np.exp(minimize_scalar(temperature_loss, bounds=(-3, 3), method="bounded").x))
    train_results = [r for r in results if records[r["index"]]["split"] == "train"]
    probe = LogisticRegression(max_iter=1000).fit(np.stack([r["embedding"] for r in train_results]), [records[r["index"]]["participant_group"] for r in train_results])
    rows = []
    retrieval = []
    for result in results:
        r = records[result["index"]]
        if r["split"] != "test":
            continue
        if args.test_domain is not None and r.get("domain") != args.test_domain:
            continue
        heatmap = result["probability"]
        truth = r["region"].astype(bool)
        pred = heatmap > threshold
        maximum = np.unravel_index(heatmap.argmax(), heatmap.shape)
        coordinates = np.argwhere(truth)
        object_pixels = np.argwhere(r["target_object"] > 0)
        diagonal = float(np.linalg.norm(np.ptp(object_pixels, axis=0) + 1)) if len(object_pixels) else float(np.hypot(*truth.shape))
        distance = float(np.linalg.norm(coordinates - maximum, axis=1).min() / diagonal) if len(coordinates) else None
        support_probability = float(torch.sigmoid(torch.tensor(result["support_logit"] / temperature)))
        actor_prediction = probe.predict(result["embedding"][None])[0]
        row = dict(id=str(r["id"]), effect=int(r["effect"]), group=str(r["group"]), domain=r.get("domain", "default"), iou=iou(pred, truth), pixel_ap=float(average_precision_score(truth.ravel(), heatmap.ravel())) if truth.any() else None, pointing_game=bool(truth[maximum]), normalized_peak_distance=distance, effect_prediction=result["effect_prediction"], support_probability=support_probability, supports=int(r["supports"]), participant_group=r["participant_group"], actor_probe=actor_prediction.item() if hasattr(actor_prediction, "item") else actor_prediction)
        for kind in ("actor", "background"):
            altered = result[kind]
            row[f"{kind}_ciou"] = iou(pred, altered > threshold) if altered is not None else None
            row[f"{kind}_iou"] = iou(altered > threshold, truth) if altered is not None else None
        candidates = [g for g in train_results if records[g["index"]]["group"] != r["group"]]
        similarities = np.array([result["embedding"] @ g["embedding"] for g in candidates])
        relevant = np.array([records[g["index"]]["effect"] == r["effect"] for g in candidates])
        retrieval.append(float(average_precision_score(relevant, similarities)) if relevant.any() else np.nan)
        if "contact" in r:
            candidates_peaks = np.argwhere((heatmap == maximum_filter(heatmap, size=9)))
            peaks = []
            for point in sorted(candidates_peaks, key=lambda point: -heatmap[tuple(point)]):
                if all(np.linalg.norm(point - other) >= 9 for other in peaks):
                    peaks.append(point)
                if len(peaks) == 5:
                    break
            distances = np.linalg.norm(np.asarray(peaks) - r["contact"], axis=1) / diagonal
            row["expert_contact_distance"] = float(distances[0])
            on_object = np.array([r["target_object"][tuple(point)] > 0 for point in peaks])
            for k in (1, 5):
                row[f"contact_hit_at_{k}"] = bool(np.any(distances[:k] <= args.contact_radius))
                row[f"proposal_success_at_{k}"] = bool(np.any((distances[:k] <= args.contact_radius) & on_object[:k]))
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No test interactions match the requested domain")
    localization = frame[frame.supports == 1]
    metrics = {k: float(localization[k].dropna().mean()) if localization[k].notna().any() else None for k in ["iou", "pixel_ap", "pointing_game", "normalized_peak_distance", "actor_ciou", "background_ciou", "actor_iou", "background_iou"]}
    metrics["effect_macro_f1"] = float(f1_score(frame.effect, frame.effect_prediction, average="macro", zero_division=0))
    metrics["same_effect_retrieval_map"] = float(np.nanmean(retrieval)) if any(np.isfinite(retrieval)) else None
    metrics["actor_probe_balanced_accuracy"] = float(balanced_accuracy_score(frame.participant_group, frame.actor_probe))
    grouped = localization.groupby(["effect", "group"]).iou.agg(["mean", "count"])
    eligible = grouped[grouped["count"] >= args.minimum_group]
    metrics["worst_group_miou"] = float(eligible["mean"].min()) if len(eligible) else None
    confidence = frame.support_probability.to_numpy()
    labels = frame.supports.to_numpy()
    metrics["effect_support_ece"] = float(sum(np.mean((confidence >= i / 15) & (confidence < (i + 1) / 15 + 1e-12)) * abs(confidence[(confidence >= i / 15) & (confidence < (i + 1) / 15 + 1e-12)].mean() - labels[(confidence >= i / 15) & (confidence < (i + 1) / 15 + 1e-12)].mean()) for i in range(15) if np.any((confidence >= i / 15) & (confidence < (i + 1) / 15 + 1e-12))))
    for column in frame:
        if column.startswith(("contact_hit", "proposal_success", "expert_contact_distance")):
            metrics[column] = float(frame[column].mean())
    metrics["actor_mask_miou_change"] = float((localization.actor_iou - localization.iou).dropna().mean()) if localization.actor_iou.notna().any() else None
    metrics["background_miou_change"] = float((localization.background_iou - localization.iou).dropna().mean()) if localization.background_iou.notna().any() else None
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), arguments=vars(args), temperature=temperature, threshold=threshold), out / "model.pt")
    with (out / "actor_probe.pkl").open("wb") as handle:
        pickle.dump(probe, handle)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    frame.to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    frame.groupby("effect").iou.mean().plot.bar(color="#4d8796")
    plt.ylabel("Effect-conditioned IoU")
    plt.tight_layout()
    plt.savefig(out / "effects.png", dpi=180)


if __name__ == "__main__":
    main()
