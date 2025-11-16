#!/usr/bin/env python3
"""
FINAL TRAINER — Unified with engine.py and model.py

Trains:
  - policy_best (CrossEntropy)
  - delta_cp
  - cp_before
  - game_result

Architecture:
  - ResNet20-like trunk (no CBAM) + Gating (same as engine uses)
  - Guaranteed weight-name compatibility with engine
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import random

from model import ResNet20_PolicyDelta   # <--- UNIFIED MODEL


# ======================================================================
# CONFIG
# ======================================================================

class Config:
    DATASET_FOLDER = Path(
        r"C:\Users\ethan\Downloads\ChessHacks\e\ChessHacks\my-chesshacks-bot\processed"
    )
    OUTPUT_DIR = Path(
        r"C:\Users\ethan\Downloads\ChessHacks\e\ChessHacks\my-chesshacks-bot\src\model_save"
    )

    EPOCHS = 10
    BATCH_SIZE = 64
    LR = 1e-3
    NUM_WORKERS = 0

    POLICY_DIM = 64 * 64 * 5

    # Delta clipping (in centipawns)
    CP_CLIP = 1500
    # Scale factor used inside the net to turn cp_scaled into "CP-like" numbers
    CP_SCALE = 200.0

    # Non-linear CP target shaping
    CP_NL_CLIP = 800.0   # clip cp_before to [-CP_NL_CLIP, CP_NL_CLIP]
    CP_NL_DIV  = 400.0   # then divide by this before tanh

    # Result discounting: game_result *= RESULT_GAMMA ** dist_to_terminal
    RESULT_GAMMA = 0.99

    # Phase shaping: how fast weights decay with distance to terminal
    PHASE_DENOM = 10.0

    # Small label noise on cp/delta (in the scaled space, i.e. after nonlinearity)
    CP_NOISE_STD = 0.02
    DELTA_NOISE_STD = 0.02

    # Delta importance threshold (in scaled units): ~0.1 ~= 20cp if CP_SCALE=200
    DELTA_IMPORTANCE_THRESHOLD = 0.1

    VAL_SPLIT = 0.1
    FLIP_PROB = 0.5

    W_DELTA = 0.5
    W_CP = 1.0
    W_RESULT = 0.25


cfg = Config()


# ======================================================================
# HELPERS
# ======================================================================

def flip_move_index(idx: int) -> int:
    """Horizontal mirror only (no color flip) on 64x64x5 move index."""
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


# ======================================================================
# DATASET
# ======================================================================

class FolderNPZDataset(Dataset):
    """
    Loads all NPZs from folder.

    Expected keys (some are optional / legacy-compatible):
      X               (N, 18, 8, 8)
      y_policy_best   (N,)   or policy_best_idx (N,)
      cp_before       (N,)
      cp_after_best   (N,)
      delta_cp        (N,)   optional (will be recomputed if missing)
      game_result     (N,)   or result (N,)
      dist_to_terminal(N,)   optional (ply distance to game end)
    """

    def __init__(self, folder, cp_clip, cp_scale, flip_prob):
        files = sorted(Path(folder).glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No NPZ files found in {folder}")

        Xs, policies, cps, deltas, results, dists = [], [], [], [], [], []

        for f in files:
            d = np.load(f)
            Xs.append(d["X"])

            # Policy index key can be y_policy_best or policy_best_idx
            if "y_policy_best" in d:
                policies.append(d["y_policy_best"])
            elif "policy_best_idx" in d:
                policies.append(d["policy_best_idx"])
            else:
                raise KeyError(f"{f} missing y_policy_best/policy_best_idx")

            cp_before = d["cp_before"].astype(np.float32)
            cp_after = d["cp_after_best"].astype(np.float32)

            # Delta
            if "delta_cp" in d:
                delta = d["delta_cp"].astype(np.float32)
            else:
                delta = cp_after - cp_before   # fallback

            delta = np.clip(delta, -cp_clip, cp_clip)

            # Game result key can be game_result or result (side-to-move POV)
            if "game_result" in d:
                game_result = d["game_result"].astype(np.float32)
            elif "result" in d:
                game_result = d["result"].astype(np.float32)
            else:
                raise KeyError(f"{f} missing game_result/result")

            # Distance to terminal (in plies), optional
            if "dist_to_terminal" in d:
                dist = d["dist_to_terminal"].astype(np.float32)
            else:
                dist = np.zeros_like(cp_before, dtype=np.float32)

            cps.append(cp_before)
            deltas.append(delta)
            results.append(game_result)
            dists.append(dist)

        self.X = np.concatenate(Xs).astype(np.float32)
        self.policy = np.concatenate(policies).astype(np.int64)
        cp = np.concatenate(cps).astype(np.float32)
        delta = np.concatenate(deltas).astype(np.float32)
        raw_result = np.concatenate(results).astype(np.float32)
        dist_to_terminal = np.concatenate(dists).astype(np.float32)

        self.dist_to_terminal = dist_to_terminal

        # ------------------------------------------------------------------
        # CP target shaping: bounded, non-linear transform into [-1, 1]
        # ------------------------------------------------------------------
        cp_bounded = np.clip(cp, -cfg.CP_NL_CLIP, cfg.CP_NL_CLIP)
        self.cp_scaled = np.tanh(cp_bounded / cfg.CP_NL_DIV).astype(np.float32)

        # ------------------------------------------------------------------
        # Delta scaling (still roughly CP / CP_SCALE, but clipped)
        # ------------------------------------------------------------------
        self.delta_scaled = (delta / cp_scale).astype(np.float32)

        # ------------------------------------------------------------------
        # Result discounting and phase factor
        # ------------------------------------------------------------------
        discount = np.power(cfg.RESULT_GAMMA, dist_to_terminal)
        self.result = (raw_result * discount).astype(np.float32)

        # Phase factor in [0,1], ~1 near terminal, smaller earlier
        self.phase = 1.0 / (1.0 + dist_to_terminal / cfg.PHASE_DENOM)
        self.phase = self.phase.astype(np.float32)

        self.flip_prob = flip_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        p = torch.from_numpy(self.X[i])
        pol = int(self.policy[i])
        cp_s = float(self.cp_scaled[i])
        delta_s = float(self.delta_scaled[i])
        res = float(self.result[i])
        phase = float(self.phase[i])

        # Flip augmentation: horizontal mirror only
        if random.random() < self.flip_prob:
            p = torch.flip(p, dims=[2])       # mirror files
            pol = flip_move_index(pol)
            # NOTE: no sign flip on delta_s — horizontal mirror
            # does not change eval or delta in side-to-move frame.

        return (
            p,
            torch.tensor(pol, dtype=torch.long),
            torch.tensor(delta_s, dtype=torch.float32),
            torch.tensor(cp_s, dtype=torch.float32),
            torch.tensor(res, dtype=torch.float32),
            torch.tensor(phase, dtype=torch.float32),
        )


# ======================================================================
# TRAIN LOOP
# ======================================================================

def run_epoch(model, loader, optimizer, device, epoch, train=True):
    model.train(train)
    ce = nn.CrossEntropyLoss()
    # We'll handle reduction manually for phase-aware weighting
    l1 = nn.SmoothL1Loss(beta=2.0, reduction="none")

    tot = tot_pol = tot_delta = tot_cp = tot_res = tot_loss = 0.0
    tag = "Train" if train else "Val"

    with tqdm(loader, ncols=100, desc=f"{tag} {epoch}") as bar:
        for planes, pol, delta_t, cp_t, res_t, phase_t in bar:
            planes = planes.to(device)
            pol = pol.to(device)
            delta_t = delta_t.to(device)
            cp_t = cp_t.to(device)
            res_t = res_t.to(device)
            phase_t = phase_t.to(device)

            # Optionally add small noise to cp/delta targets (train only)
            if train:
                if cfg.CP_NOISE_STD > 0.0:
                    cp_t_used = cp_t + torch.randn_like(cp_t) * cfg.CP_NOISE_STD
                else:
                    cp_t_used = cp_t

                if cfg.DELTA_NOISE_STD > 0.0:
                    delta_t_used = delta_t + torch.randn_like(delta_t) * cfg.DELTA_NOISE_STD
                else:
                    delta_t_used = delta_t
            else:
                cp_t_used = cp_t
                delta_t_used = delta_t

            with torch.set_grad_enabled(train):
                (
                    logits,
                    delta_real,
                    delta_pred,
                    cp_real,
                    cp_pred,
                    res_pred,
                ) = model(planes)

                # Policy loss (no label smoothing here for simplicity)
                loss_pol = ce(logits, pol)

                # Per-sample scalar losses
                delta_loss_vec = l1(delta_pred, delta_t_used)   # [B]
                cp_loss_vec    = l1(cp_pred, cp_t_used)         # [B]
                res_loss_vec   = l1(res_pred, res_t)            # [B]

                # ------------------------------------------------------------------
                # Phase-aware and "interestingness"-aware weighting
                # ------------------------------------------------------------------
                # phase_t ~ 1 near terminal, smaller earlier
                w_res = 0.1 + 0.9 * phase_t
                w_cp  = 0.5 + 0.5 * phase_t

                # More weight for positions where |delta| is non-trivial
                delta_mag = delta_t.abs()
                thr = cfg.DELTA_IMPORTANCE_THRESHOLD
                interesting = torch.clamp(delta_mag / thr, max=1.0)
                w_delta_phase = 0.3 + 0.7 * phase_t
                w_delta = w_delta_phase * interesting

                loss_delta = (w_delta * delta_loss_vec).mean()
                loss_cp    = (w_cp * cp_loss_vec).mean()
                loss_res   = (w_res * res_loss_vec).mean()

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
            tot += bs
            tot_pol += loss_pol.item() * bs
            tot_delta += loss_delta.item() * bs
            tot_cp += loss_cp.item() * bs
            tot_res += loss_res.item() * bs
            tot_loss += loss.item() * bs

            bar.set_postfix(
                {
                    "pol": f"{tot_pol / tot:.3f}",
                    "dlt": f"{tot_delta / tot:.3f}",
                    "cp": f"{tot_cp / tot:.3f}",
                    "res": f"{tot_res / tot:.3f}",
                    "tot": f"{tot_loss / tot:.3f}",
                }
            )

    return tot_loss / tot


# ======================================================================
# MAIN
# ======================================================================

def main():
    ds = FolderNPZDataset(
        cfg.DATASET_FOLDER, cfg.CP_CLIP, cfg.CP_SCALE, cfg.FLIP_PROB
    )
    print("Dataset size:", len(ds))

    # Split
    val_size = int(len(ds) * cfg.VAL_SPLIT)
    train_set, val_set = random_split(ds, [len(ds) - val_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=cfg.BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.BATCH_SIZE, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet20_PolicyDelta(
        in_planes=ds.X.shape[1],
        policy_dim=cfg.POLICY_DIM,
        cp_scale=cfg.CP_SCALE,
    ).to(device)

    print("TRAINER — Using in_planes:", ds.X.shape[1])

    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LR * 0.1
    )

    out = Path(cfg.OUTPUT_DIR)
    out.mkdir(exist_ok=True, parents=True)

    best = float("inf")

    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, device, epoch, train=True
        )
        val_loss = run_epoch(
            model, val_loader, optimizer, device, epoch, train=False
        )
        scheduler.step()

        print(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")

        torch.save(model.state_dict(), out / "last.pt")

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), out / "best.pt")
            print(f"*** NEW BEST ({val_loss:.4f}) ***")

    print("Training complete.")


if __name__ == "__main__":
    main()
