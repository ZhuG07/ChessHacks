"""
Neural network definitions for the engine.

Two architectures exist in this project, and both share the same input
encoding (18 x 8 x 8 planes) and the same four output heads:

  * ResNet20_PolicyDelta   - 6 plain residual blocks at 128 channels.
                             This is the smaller net used by the self-play
                             fine-tuning loop (best_selfplay.pt).
  * ResNetCBAM_PolicyDelta - 9 residual blocks at 256 channels, each with a
                             CBAM (channel + spatial) attention module.
                             This is the larger supervised net from train.py
                             (best.pt).

Both forward() methods return the same 6-tuple:

    policy_logits, delta_real, delta_scaled, cp_real, cp_scaled, result

`load_model()` reads a checkpoint, works out which architecture (and which
width / depth) it was trained with, and returns a ready-to-use model in eval
mode. That way the engine never has to hardcode which file goes with which
class.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


# ============================================================
# Plain residual block
# ============================================================

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # Match dimensions for the residual path if needed
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.relu(h + self.shortcut(x))


# ============================================================
# CBAM attention (Woo et al., 2018)
# ============================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        mx = self.max_pool(x).view(b, c)
        attn = self.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * s


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class CBAMBlock(nn.Module):
    """Residual block with CBAM applied to the residual branch."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # Parameter-free when shapes match; named `short` to stay
        # compatible with checkpoints written by train.py.
        self.short = nn.Sequential()
        if in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        self.cbam = CBAM(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = self.cbam(h)
        return self.relu(h + self.short(x))


# ============================================================
# Shared trunk + heads
# ============================================================

class _PolicyValueNet(nn.Module):
    """
    Stem -> residual trunk -> global average pool -> shared FC with a
    sigmoid gate -> four heads.

    Heads:
      * policy_logits : one logit per (from, to, promotion) triple
      * delta_cp      : predicted eval change from playing the best move
      * cp            : absolute evaluation, side-to-move point of view
      * result        : expected game result in [-1, 1] (tanh)
    """

    block_cls: type[nn.Module] = BasicBlock

    def __init__(self, in_planes=18, policy_dim=20480, cp_scale=200.0,
                 trunk_channels=128, num_blocks=6):
        super().__init__()
        self.cp_scale = cp_scale

        self.conv_in = nn.Conv2d(in_planes, trunk_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in   = nn.BatchNorm2d(trunk_channels)
        self.relu    = nn.ReLU(inplace=True)

        self.trunk = nn.Sequential(
            *[self.block_cls(trunk_channels, trunk_channels) for _ in range(num_blocks)]
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_shared = nn.Linear(trunk_channels, 256)
        self.gate = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.Sigmoid(),
        )

        self.policy_head = nn.Linear(256, policy_dim)
        self.delta_head  = nn.Linear(256, 1)
        self.cp_head     = nn.Linear(256, 1)
        self.result_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.trunk(x)
        x = self.avg(x).view(x.size(0), -1)

        x = self.fc_shared(x)
        x = self.relu(x * self.gate(x))

        policy_logits = self.policy_head(x)

        delta_scaled = self.delta_head(x).squeeze(-1)
        cp_scaled    = self.cp_head(x).squeeze(-1)
        result       = torch.tanh(self.result_head(x).squeeze(-1))

        delta_real = delta_scaled * self.cp_scale
        cp_real    = cp_scaled * self.cp_scale

        return policy_logits, delta_real, delta_scaled, cp_real, cp_scaled, result


class ResNet20_PolicyDelta(_PolicyValueNet):
    """Plain residual net (default 128 channels x 6 blocks)."""

    block_cls = BasicBlock


class ResNetCBAM_PolicyDelta(_PolicyValueNet):
    """Residual net with CBAM attention (default 256 channels x 9 blocks)."""

    block_cls = CBAMBlock

    def __init__(self, in_planes=18, policy_dim=20480, cp_scale=200.0,
                 trunk_channels=256, num_blocks=9):
        super().__init__(in_planes, policy_dim, cp_scale, trunk_channels, num_blocks)


# ============================================================
# Checkpoint loading
# ============================================================

def _extract_state_dict(obj) -> dict:
    """Accept either a raw state_dict or a {'model': state_dict, ...} bundle."""
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def describe_state_dict(state: dict) -> dict:
    """Infer architecture hyperparameters from parameter names and shapes."""
    conv_in = state["conv_in.weight"]
    trunk_ids = {int(k.split(".")[1]) for k in state if k.startswith("trunk.")}
    return {
        "arch": "cbam" if any(".cbam." in k for k in state) else "plain",
        "in_planes": conv_in.shape[1],
        "trunk_channels": conv_in.shape[0],
        "num_blocks": max(trunk_ids) + 1 if trunk_ids else 0,
        "policy_dim": state["policy_head.weight"].shape[0],
    }


def load_model(path: str | Path, device: str | torch.device = "cpu",
               cp_scale: float = 200.0) -> tuple[nn.Module, dict]:
    """
    Build the right network for a checkpoint and load its weights.

    Returns (model, info) where info describes the detected architecture.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        head = fh.read(64)
    if head.startswith(b"version https://git-lfs"):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not the real weights. "
            "Run `git lfs pull` or `python scripts/fetch_weights.py`."
        )

    state = _extract_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    info = describe_state_dict(state)

    cls = ResNetCBAM_PolicyDelta if info["arch"] == "cbam" else ResNet20_PolicyDelta
    model = cls(
        in_planes=info["in_planes"],
        policy_dim=info["policy_dim"],
        cp_scale=cp_scale,
        trunk_channels=info["trunk_channels"],
        num_blocks=info["num_blocks"],
    )
    model.load_state_dict(state)
    model.to(device).eval()

    info["parameters"] = sum(p.numel() for p in model.parameters())
    info["checkpoint"] = path.name
    return model, info
