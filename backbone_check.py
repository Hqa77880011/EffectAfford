"""One generated example through the real pretrained ViT-B/16, on CPU."""
import json
import torch

from effectafford import EffectAfford


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = EffectAfford(2, "full", backbone="vit_b_16")
    pre, post = torch.rand(1, 1, 3, 224, 224), torch.rand(1, 1, 3, 224, 224)
    target, support = torch.rand(1, 3, 224, 224), torch.ones(1, 1, 224, 224)
    embedding = model.embed(pre, post, support)
    logits, confidence = model.ground(embedding, target)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)) + confidence.square().mean()
    loss.backward()
    parameters = [model.encoder.vit.conv_proj.weight, model.encoder.projection.weight, model.change[-1].weight, model.support.weight]
    assert torch.isfinite(loss)
    assert all(value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0 for value in parameters)
    before = parameters[0].detach().clone()
    torch.optim.SGD(parameters, lr=.001).step()
    assert not torch.equal(before, parameters[0])
    print(json.dumps({"pretrained": "torchvision ViT_B_16_Weights.IMAGENET1K_V1", "input": [1, 1, 3, 224, 224], "loss": float(loss.detach()), "backbone_head_gradients": True, "patch_weight_update": True}))


if __name__ == "__main__":
    main()
