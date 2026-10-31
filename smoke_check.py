"""Generated-window functional test using the explicit small-CNN control."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from send2trash import send2trash

import effectafford as implementation


def run():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    work = Path(tempfile.mkdtemp(prefix="effectafford-check-"))
    try:
        rng = np.random.default_rng(0)
        rows = []
        for index, split in enumerate(["train"] * 6 + ["val"] * 2 + ["test"] * 2):
            pre = rng.integers(0, 255, (8, 32, 32, 3), dtype=np.uint8)
            post = rng.integers(0, 255, pre.shape, dtype=np.uint8)
            mask = np.zeros((32, 32), np.uint8)
            mask[8:24, 8:24] = 1
            np.savez(work / f"{index}.npz", pre=pre, post=post, target=pre[0], source_object=mask, target_object=mask, actor_mask=np.zeros_like(mask), region=mask, contact=np.array([16, 16]))
            rows.append(dict(id=str(index), path=f"{index}.npz", split=split, effect=index % 2, group=str(index), verb="touch", supports=1, participant_group=index % 2))
        manifest = work / "interactions.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        model = implementation.EffectAfford(2, "full")
        records = [implementation.load_record(work / f"{i}.npz") for i in range(6)]
        pre, post, target, support, truth = implementation.batch(records)
        embedding = model.embed(pre, post, support)
        logits, confidence = model.ground(embedding, target)
        eq, hard = implementation.contrastive(embedding, np.array([0, 1, 0, 1, 0, 1]), np.arange(6), np.array(["touch"] * 6), "full")
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, truth) + confidence.square().mean() + eq + hard
        loss.backward()
        assert torch.isfinite(loss)
        for module in (model.change[-1], model.query, model.key, model.support):
            assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all() and module.weight.grad.abs().sum() > 0
        before = model.change[-1].weight.detach().clone()
        torch.optim.Adam(model.parameters(), lr=.001).step()
        assert not torch.equal(before, model.change[-1].weight)
        model.eval()
        with torch.no_grad():
            expected = model.ground(model.embed(pre, post, support), target)[0]
        torch.save(model.state_dict(), work / "state.pt")
        restored = implementation.EffectAfford(2, "full").eval()
        restored.load_state_dict(torch.load(work / "state.pt", weights_only=True))
        with torch.no_grad():
            torch.testing.assert_close(expected, restored.ground(restored.embed(pre, post, support), target)[0], atol=1e-6, rtol=1e-6)
        assert implementation.js_heatmap(logits, logits, torch.ones_like(logits)).abs().max() < 1e-7
        sys.argv = ["effectafford", str(manifest), "--backbone", "cnn", "--classes", "2", "--epochs", "2", "--batch-size", "6", "--warmup-epochs", "1", "--patience", "1", "--minimum-group", "1", "--output", str(work / "run")]
        implementation.main()
        history = pd.read_csv(work / "run/history.csv")
        assert {"validation_miou", "validation_retrieval_map", "selection_score", "backbone_lr", "head_lr"} <= set(history)
        selected = torch.load(work / "run/model.pt", weights_only=True)
        assert 1 <= selected["arguments"]["selected_epoch"] <= len(history)
        assert len(implementation.sample_batch({0: {"a": [0], "b": [1], "c": [2]}, 1: {"d": [3], "e": [4], "f": [5]}}, np.random.default_rng(0), 64)) == 64
        metrics = json.loads((work / "run/metrics.json").read_text())
        assert {"iou", "pixel_ap", "pointing_game", "same_effect_retrieval_map", "effect_support_ece", "contact_hit_at_5"} <= metrics.keys()
        sys.argv = ["effectafford", str(manifest), "--checkpoint", str(work / "run/model.pt"), "--minimum-group", "1", "--output", str(work / "reload")]
        implementation.main()
        repeated = json.loads((work / "reload/metrics.json").read_text())
        assert metrics == repeated
        print(json.dumps({"loss": float(loss.detach()), "eight_frame_windows": True, "gradient_update": True, "roundtrip": True, "frozen_checkpoint_evaluation": True, "identical_heatmap_js": 0.}))
    finally:
        send2trash(str(work))


if __name__ == "__main__":
    run()
