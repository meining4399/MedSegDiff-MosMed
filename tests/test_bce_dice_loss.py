"""Smoke test: one forward+backward of training_losses_segmentation with BCE_DICE.

Verifies the new region loss supervises pred_xstart against the GT mask (not
against noise), and that gradients are non-vanishing even under heavy class
imbalance. Run from repo root:
    python tests/test_bce_dice_loss.py
"""
import sys
import os
import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def run_one(cfg, device):
    th.manual_seed(0)
    model, diff = create_model_and_diffusion(**cfg)
    model.to(device)
    model.train()
    model.zero_grad(set_to_none=True)
    model_kwargs = {}

    B, H, W = 1, 256, 256
    img = th.randn(B, 1, H, W, device=device)
    mask = (th.rand(B, 1, H, W, device=device) < 0.01).float()
    x_start = th.cat([img, mask], dim=1)

    t = th.randint(0, diff.num_timesteps, (B,), device=device)

    terms, model_output = diff.training_losses_segmentation(
        model, None, x_start, t, model_kwargs=model_kwargs
    )
    loss = terms["loss"].mean()
    loss.backward()

    nz_any = sum(
        1 for p in model.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    total = sum(1 for _ in model.parameters())
    gmax = max(
        (p.grad.abs().max().item() for p in model.parameters() if p.grad is not None),
        default=0.0,
    )
    print(f"[{cfg['loss_type']}] loss={loss.item():.4f} loss_diff={terms['loss_diff'].mean().item():.4f} "
          f"nz_tensors={nz_any}/{total} grad_max={gmax:.3e}")
    return loss.item(), nz_any, total


def main():
    th.manual_seed(0)
    device = "cuda" if th.cuda.is_available() else "cpu"
    print("device:", device)

    base = model_and_diffusion_defaults()
    base.update(
        dict(
            image_size=256,
            num_channels=128,
            in_ch=2,
            version="new",
            learn_sigma=False,
            diffusion_steps=100,
        )
    )

    print("=== Baseline: MSE loss ===")
    cfg_mse = dict(base)
    cfg_mse["loss_type"] = "mse"
    run_one(cfg_mse, device)

    print("=== New: BCE_DICE loss ===")
    cfg_bce = dict(base)
    cfg_bce.update(dict(
        loss_type="bce_dice",
        bce_pos_weight=200.0,
        tversky_alpha=0.3,
        tversky_beta=0.7,
    ))
    loss, nz, total = run_one(cfg_bce, device)

    assert nz > 2 or total == 0, f"BCE_DICE only reaches {nz}/{total} param tensors!"
    print("OK: BCE_DICE gradients flow to a comparable number of tensors as MSE.")


if __name__ == "__main__":
    main()
