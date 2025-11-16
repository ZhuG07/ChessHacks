#!/usr/bin/env python3
"""
TRAINER — Delta-CP + Absolute CP + Result + ResNet20 + Board Flipping Augmentation + Folder Dataset
===================================================================================================

This trainer:
  ✔ Predicts delta_cp (local improvement)
  ✔ Predicts cp_before (absolute evaluation)
  ✔ Predicts game_result (POV of side to move, in [-1, 0, 1])
  ✔ Uses stronger ResNet-20 (256 channels)
  ✔ Adds board-flip augmentation (horizontal + color flip)
  ✔ Folder of NPZ files supported
  ✔ TQDM progress
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import random


# ============================================================================
# CONFIG
# ============================================================================

class Config:
    DATASET_FOLDER = r"training\whiteNoise/processed"
    OUTPUT_DIR = "checkpoints_resnet20_trained_models"
    EPOCHS = 20
    BATCH_SIZE = 64
    LR = 3e-4
    NUM_WORKERS = 0

    POLICY_DIM = 64 * 64 * 5
    CP_CLIP = 1500
    CP_SCALE = 200.0

    VAL_SPLIT = 0.1
    FLIP_PROB = 0.5      # flip 50% of training samples

    # Loss weights
    W_DELTA = 1.0
    W_CP = 1.0
    W_RESULT = 0.5


cfg = Config()


# ============================================================================
# POLICY INDEX MAP FOR FLIPPED BOARD
# ============================================================================

PROMO_PIECES = [None, 5, 4, 3, 2]  # Q,R,B,N etc.

def flip_move_index(idx):
    """
    Computes new policy index after horizontal + perspective flip
    """
    from_sq = idx // (64 * 5)
    rest = idx % (64 * 5)
    to_sq = rest // 5
    promo_idx = rest % 5

    f_file = from_sq % 8
    f_rank = from_sq // 8
    t_file = to_sq % 8
    t_rank = to_sq // 8

    f_file_f = 7 - f_file
    t_file_f = 7 - t_file

    new_from = f_rank * 8 + f_file_f
    new_to = t_rank * 8 + t_file_f

    return (new_from * 64 + new_to) * 5 + promo_idx


# ============================================================================
# RESNET-20 WITH POLICY + DELTA + CP + RESULT HEADS
# ============================================================================
# ============================================================================
# RESNET-CBAM + GLOBAL GATING WITH POLICY + DELTA + CP + RESULT HEADS
# ============================================================================

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
    We keep channels constant (e.g. 256) throughout the trunk.
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
    Best-architecture version:

      - Input conv to 256 channels
      - 9 × CBAM residual blocks at 256 channels
      - Global average pool → 256-d feature
      - Shared FC → 256-d
      - Global gating MLP on shared features
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
        trunk_channels = 256

        # Stem
        self.conv_in = nn.Conv2d(in_planes, trunk_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in   = nn.BatchNorm2d(trunk_channels)
        self.relu    = nn.ReLU(inplace=True)

        # Residual trunk: 9 CBAM blocks, all 256 channels
        blocks = []
        for _ in range(9):
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



# ============================================================================
# DATASET + BOARD FLIPPING AUGMENTATION
# ============================================================================

class FolderNPZDataset(Dataset):
    """
    Loads all NPZ files in a folder.

    Expected keys per NPZ (from your self-play generator):
      - X              : (N, 18, 8, 8)
      - y_policy_best  : (N,)
      - cp_before      : (N,)
      - cp_after_best  : (N,)
      - delta_cp       : (N,)   (optional; recomputed if missing)
      - game_result    : (N,)   in {-1,0,1}
    """

    def __init__(self, folder, cp_clip, cp_scale, flip_prob):
        files = sorted(Path(folder).glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No NPZs found in {folder}")

        Xs = []
        policies = []
        deltas = []
        cps = []
        results = []

        for f in files:
            d = np.load(f)
            Xs.append(d["X"])
            policies.append(d["y_policy_best"])

            cp_before = d["cp_before"].astype(np.float32)
            cp_after = d["cp_after_best"].astype(np.float32)

            if "delta_cp" in d:
                delta_cp = d["delta_cp"].astype(np.float32)
            else:
                delta_cp = cp_after - cp_before

            delta_cp = np.clip(delta_cp, -cp_clip, cp_clip)

            game_result = d["game_result"].astype(np.float32)  # -1,0,1 POV of side to move

            cps.append(cp_before)
            deltas.append(delta_cp)
            results.append(game_result)

        self.X = np.concatenate(Xs).astype(np.float32)
        self.policy = np.concatenate(policies).astype(np.int64)

        cp = np.concatenate(cps).astype(np.float32)
        delta = np.concatenate(deltas).astype(np.float32)
        result = np.concatenate(results).astype(np.float32)

        self.cp_scaled = cp / cp_scale
        self.delta_scaled = delta / cp_scale
        self.result = result

        self.flip_prob = flip_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        p = torch.from_numpy(self.X[idx])
        pol = int(self.policy[idx])
        delta = float(self.delta_scaled[idx])
        cp = float(self.cp_scaled[idx])
        res = float(self.result[idx])

        # Board flip augmentation
        if random.random() < self.flip_prob:
            p = torch.flip(p, dims=[2])   # flip horizontally (file-wise)
            pol = flip_move_index(pol)
            delta = -delta                # perspective flips
            cp = cp                       # cp_before does NOT change sign under horizontal mirror
            res = res                     # result unchanged under mirror

        return (
            p,
            torch.tensor(pol, dtype=torch.long),
            torch.tensor(delta, dtype=torch.float32),
            torch.tensor(cp, dtype=torch.float32),
            torch.tensor(res, dtype=torch.float32),
        )


# ============================================================================
# TRAIN/VAL LOOP (PRINTS LOSSES)
# ============================================================================

def run_epoch(model, loader, optimizer, device, epoch, train=True):
    model.train(train)
    ce = nn.CrossEntropyLoss()
    l1 = nn.SmoothL1Loss(beta=2.0)

    total_loss = total_pol = total_delta = total_cp = total_res = 0.0
    total = 0

    tag = "Train" if train else "Val"
    with tqdm(loader, desc=f"{tag} {epoch}", ncols=100) as t:
        for planes, pol, delta_t, cp_t, res_t in t:
            planes = planes.to(device)
            pol = pol.to(device)
            delta_t = delta_t.to(device)
            cp_t = cp_t.to(device)
            res_t = res_t.to(device)

            with torch.set_grad_enabled(train):
                (
                    logits,
                    _delta_real,
                    delta_pred_scaled,
                    _cp_real,
                    cp_pred_scaled,
                    res_pred,
                ) = model(planes)

                loss_pol = ce(logits, pol)
                loss_delta = l1(delta_pred_scaled, delta_t)
                loss_cp = l1(cp_pred_scaled, cp_t)
                loss_res = l1(res_pred, res_t)

                loss = (
                    loss_pol
                    + cfg.W_DELTA * loss_delta
                    + cfg.W_CP * loss_cp
                    + cfg.W_RESULT * loss_res
                )

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            bs = planes.size(0)
            total += bs
            total_pol += loss_pol.item() * bs
            total_delta += loss_delta.item() * bs
            total_cp += loss_cp.item() * bs
            total_res += loss_res.item() * bs
            total_loss += loss.item() * bs

            t.set_postfix({
                "pol": f"{total_pol / total:.3f}",
                "dlt": f"{total_delta / total:.3f}",
                "cp": f"{total_cp / total:.3f}",
                "res": f"{total_res / total:.3f}",
                "tot": f"{total_loss / total:.3f}",
            })

    return total_loss / total


# ============================================================================
# MAIN
# ============================================================================

def main():
    ds = FolderNPZDataset(cfg.DATASET_FOLDER, cfg.CP_CLIP, cfg.CP_SCALE, cfg.FLIP_PROB)
    N = len(ds)
    print("Dataset size: ", N)

    val_size = int(N * cfg.VAL_SPLIT)
    train_set, val_set = random_split(ds, [N - val_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("(you want this to be cuda btw) Currently using :" , device)
    model = ResNet20_PolicyDelta(ds.X.shape[1], cfg.POLICY_DIM, cfg.CP_SCALE).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.LR)

    out = Path(cfg.OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    best = float("inf")

    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, opt, device, epoch, train=True)
        val_loss = run_epoch(model, val_loader, opt, device, epoch, train=False)

        print(f"\nEpoch {epoch} summary: train={train_loss:.4f} val={val_loss:.4f}\n")

        torch.save(model.state_dict(), out / "last.pt")

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), out / "best.pt")
            print(f"**** NEW BEST at epoch {epoch}, val={val_loss:.4f} ****\n")


if __name__ == "__main__":
    main()
