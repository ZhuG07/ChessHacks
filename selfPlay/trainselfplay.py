# training/selfplay/train_selfplay_from_npz.py

from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .config import (
    MODEL_PATH,
    SELFPLAY_OUT_DIR,
    NUM_PLANES,
    POLICY_DIM,
    CP_SCALE,
)

# We can either import ResNet20_PolicyDelta from your engine or re-define it here.
# To avoid circular imports in training mode, it's often cleaner to re-define.
# If you prefer importing, replace this with:
# from engine import ResNet20_PolicyDelta

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
        maxv = self.max_pool(x).view(b, c)

        attn = self.mlp(avg) + self.mlp(maxv)
        attn = self.sigmoid(attn).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        avg = torch.mean(x, dim=1, keepdim=True)          # (B, 1, H, W)
        maxv, _ = torch.max(x, dim=1, keepdim=True)       # (B, 1, H, W)
        s = torch.cat([avg, maxv], dim=1)                 # (B, 2, H, W)
        s = self.conv(s)
        s = self.sigmoid(s)
        return x * s


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class BasicBlock(nn.Module):
    """
    Residual block with CBAM attention.

    Keeps spatial size the same (stride=1 always here).
    We keep channels constant (e.g. 128) throughout the trunk.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

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
        h = self.cbam(h)              # attention over features
        h = h + self.short(x)         # residual connection
        return self.relu(h)


class ResNet20_PolicyDelta(nn.Module):
    """
    Lite best-architecture version:

      - Input conv to 128 channels
      - 6 × CBAM residual blocks at 128 channels
      - Global average pool → C-d feature
      - Shared FC → 256-d
      - Global gating MLP on shared 256-d feature
      - Heads:
          * policy_head: logits over policy_dim
          * delta_head : scaled delta_cp
          * cp_head    : scaled cp_before
          * result_head: tanh in [-1, 1]

    Forward return (for compatibility with your training loop):
      policy_logits, delta_real, delta_scaled, cp_real, cp_scaled, result
    """

    def __init__(self, in_planes, policy_dim, cp_scale):
        super().__init__()

        self.cp_scale = cp_scale
        trunk_channels = 128   # ↓ narrower trunk for speed

        # Stem
        self.conv_in = nn.Conv2d(in_planes, trunk_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in   = nn.BatchNorm2d(trunk_channels)
        self.relu    = nn.ReLU(inplace=True)

        # Residual trunk: 6 CBAM blocks, all 128 channels
        blocks = []
        for _ in range(6):     # ↓ fewer blocks for speed
            blocks.append(BasicBlock(trunk_channels, trunk_channels))
        self.trunk = nn.Sequential(*blocks)

        # Global pooling
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        # Shared feature projection
        self.fc_shared = nn.Linear(trunk_channels, 256)

        # Global context gating on shared 256-d feature
        self.gate = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.Sigmoid(),
        )

        # Heads
        self.policy_head = nn.Linear(256, policy_dim)
        self.delta_head  = nn.Linear(256, 1)   # scaled delta_cp
        self.cp_head     = nn.Linear(256, 1)   # scaled cp_before
        self.result_head = nn.Linear(256, 1)   # game result in [-1,1] via tanh

    def forward(self, x):
        # Stem + trunk
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.trunk(x)                           # (B, C, 8, 8)

        # Global pooling
        x = self.avg(x)                             # (B, C, 1, 1)
        x = x.view(x.size(0), -1)                   # (B, C)

        # Shared feature
        x = self.fc_shared(x)                       # (B, 256)

        # Global context gate
        gates = self.gate(x)                        # (B, 256) in (0,1)
        x = x * gates                               # gated shared feature
        x = self.relu(x)

        # Policy logits
        policy_logits = self.policy_head(x)         # (B, policy_dim)

        # Scaled evals
        delta_scaled = self.delta_head(x).squeeze(-1)  # (B,)
        cp_scaled    = self.cp_head(x).squeeze(-1)     # (B,)

        # Convert to real cp units
        delta_real = delta_scaled * self.cp_scale
        cp_real    = cp_scaled * self.cp_scale

        # Result in [-1, 1]
        result_raw = self.result_head(x).squeeze(-1)
        result = torch.tanh(result_raw)

        return policy_logits, delta_real, delta_scaled, cp_real, cp_scaled, result




class SelfPlayNPZDataset(Dataset):
    def __init__(self, npz_path: Path, cp_scale: float = CP_SCALE):
        data = np.load(npz_path)
        self.X = data["X"]  # (N, 18, 8, 8)
        self.policy_best_idx = data["policy_best_idx"]  # (N,)
        self.cp_before = data["cp_before"]  # (N,)
        self.delta_cp = data["delta_cp"]  # (N,)
        self.result = data["result"]  # (N,)
        self.cp_scale = cp_scale

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        policy_idx = int(self.policy_best_idx[idx])
        cp_before = float(self.cp_before[idx])
        delta_cp = float(self.delta_cp[idx])
        result = float(self.result[idx])  # [-1,1]

        # scale cp targets consistently with model
        cp_target_scaled = cp_before / self.cp_scale
        delta_target_scaled = delta_cp / self.cp_scale

        return (
            torch.from_numpy(x),         # (18,8,8)
            torch.tensor(policy_idx),    # ()
            torch.tensor(cp_target_scaled, dtype=torch.float32),
            torch.tensor(delta_target_scaled, dtype=torch.float32),
            torch.tensor(result, dtype=torch.float32),
        )


def train_selfplay_model(
    npz_path: Path,
    out_model_path: Path,
    batch_size: int = 256,
    epochs: int = 5,
    lr: float = 1e-4,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = SelfPlayNPZDataset(npz_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = ResNet20_PolicyDelta(
        planes=NUM_PLANES,
        policy_dim=POLICY_DIM,
        cp_scale=CP_SCALE,
    ).to(device)

    # Load starting weights
    if MODEL_PATH.is_file():
        print(f"[TRAIN] Loading initial weights from {MODEL_PATH}")
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state, strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        total_policy_loss = 0.0
        total_cp_loss = 0.0
        total_delta_loss = 0.0
        total_result_loss = 0.0
        count = 0

        for batch in dl:
            x, policy_idx, cp_target_scaled, delta_target_scaled, result_target = batch
            x = x.to(device)  # (B,18,8,8)
            policy_idx = policy_idx.to(device)
            cp_target_scaled = cp_target_scaled.to(device)
            delta_target_scaled = delta_target_scaled.to(device)
            result_target = result_target.to(device)

            (
                policy_logits,
                delta_real,
                delta_scaled_pred,
                cp_real,
                cp_scaled_pred,
                result_pred,
            ) = model(x)

            # Policy loss: cross-entropy over POLICY_DIM vs policy_idx
            policy_loss = ce_loss(policy_logits, policy_idx)

            # cp loss: MSE in scaled space
            cp_loss = mse_loss(cp_scaled_pred, cp_target_scaled)

            # delta loss: MSE in scaled space
            delta_loss = mse_loss(delta_scaled_pred, delta_target_scaled)

            # result loss: MSE in [-1,1]
            result_loss = mse_loss(result_pred, result_target)

            loss = policy_loss + cp_loss + delta_loss + result_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            B = x.size(0)
            total_loss += float(loss.item()) * B
            total_policy_loss += float(policy_loss.item()) * B
            total_cp_loss += float(cp_loss.item()) * B
            total_delta_loss += float(delta_loss.item()) * B
            total_result_loss += float(result_loss.item()) * B
            count += B

        print(
            f"[EPOCH {epoch+1}/{epochs}] "
            f"loss={total_loss / count:.4f}  "
            f"policy={total_policy_loss / count:.4f}  "
            f"cp={total_cp_loss / count:.4f}  "
            f"delta={total_delta_loss / count:.4f}  "
            f"result={total_result_loss / count:.4f}"
        )

    print(f"[TRAIN] Saving fine-tuned model to {out_model_path}")
    torch.save(model.state_dict(), out_model_path)


if __name__ == "__main__":
    npz_file = SELFPLAY_OUT_DIR / "selfplay_sf_topk_dataset.npz"
    out_model = MODEL_PATH.parent / "best_selfplay.pt"
    train_selfplay_model(npz_file, out_model)
