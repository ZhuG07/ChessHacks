#!/usr/bin/env python3
"""
Simple inference test for your trained chess model (.pt)
using one NPZ shard from your Stockfish/self-play generator.

No argparse — just edit the variables below.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# CONFIG — EDIT THESE THREE LINES
# ============================================================
MODEL_PATH = r"C:\Users\ethan\Downloads\ChessHacks\ChessHacks2025\training\whiteNoise\checkpoints_delta_resnet20\best.pt"
    # path to your model .pt
NPZ_PATH   = r"C:\Users\ethan\Downloads\ChessHacks\ChessHacks2025\training\whiteNoise\processed\sf_supervised_dataset.npz"  # path to your NPZ file
DEVICE     = "cuda"                               # "cuda" or "cpu"

BATCH_SIZE = 256
MAX_SAMPLES = 50000    # max positions to load from NPZ



# ============================================================
# DATASET
# ============================================================
class NPZChessDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path: str, max_samples: int | None = None):
        data = np.load(npz_path)

        self.X = data["X"].astype(np.float32)
        self.y_policy = data["y_policy_best"].astype(np.int64)

        # True delta_cp = cp_after - cp_before
        cp_before = data["cp_before"].astype(np.float32)
        cp_after = data["cp_after_best"].astype(np.float32)
        delta_cp = cp_after - cp_before

        # Use same scaling as training
        self.delta_scaled = delta_cp / 200.0   # CP_SCALE = 200.0

        if max_samples is not None:
            n = min(max_samples, self.X.shape[0])
            self.X = self.X[:n]
            self.y_policy = self.y_policy[:n]
            self.delta_scaled = self.delta_scaled[:n]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y_pol = torch.tensor(self.y_policy[idx], dtype=torch.long)
        y_delta = torch.tensor(self.delta_scaled[idx], dtype=torch.float32)
        return x, y_pol, y_delta



# ============================================================
# MODEL LOADING
# ============================================================
def load_model(path, device):
    # Import your model class from the trainer file
    from train import ResNet20_PolicyDelta

    # IMPORTANT — match the parameters used during training
    PLANES = 18                 # ds.X.shape[1] = 18 planes
    POLICY_DIM = 64 * 64 * 5    # 20480
    CP_SCALE = 200.0            # same as cfg.CP_SCALE

    # Recreate the architecture
    model = ResNet20_PolicyDelta(PLANES, POLICY_DIM, CP_SCALE).to(device)

    # Load state_dict
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)

    model.eval()
    return model


# ============================================================
# FORWARD EXTRACTION
# ============================================================

def run_forward(model, x):
    """
    Modify this if your model returns extra outputs.
    """
    out = model(x)

    if isinstance(out, torch.Tensor):
        return out

    if isinstance(out, (tuple, list)):
        return out[0]

    if isinstance(out, dict):
        if "policy_logits" not in out:
            raise KeyError("Model dict missing 'policy_logits'")
        return out["policy_logits"]

    raise TypeError(f"Unexpected model output: {type(out)}")



# ============================================================
# EVALUATION LOOP
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device):
    import torch.nn.functional as F

    total_ce = 0.0
    total_acc = 0
    total = 0

    mse = 0.0
    mae = 0.0

    for x, y_policy, y_delta in loader:
        x = x.to(device)
        y_policy = y_policy.to(device)
        y_delta = y_delta.to(device)   # scaled delta

        logits, delta_real, delta_scaled = model(x)

        # --- Policy metrics ---
        ce = F.cross_entropy(logits, y_policy, reduction="sum")
        preds = logits.argmax(dim=1)

        total_ce += ce.item()
        total_acc += (preds == y_policy).sum().item()

        # --- Delta (eval bar change) metrics ---
        # y_delta is scaled, delta_real is unscaled
        # true_real = y_delta * CP_SCALE
        true_real = y_delta * 200.0

        mse += ((delta_real - true_real) ** 2).sum().item()
        mae += (delta_real - true_real).abs().sum().item()

        total += y_policy.size(0)

    return {
        "ce": total_ce / total,
        "acc": total_acc / total,
        "mse": mse / total,
        "rmse": (mse / total) ** 0.5,
        "mae": mae / total,
        "samples": total,
    }



# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH, device)

    print(f"Loading dataset: {NPZ_PATH}")
    dataset = NPZChessDataset(NPZ_PATH, MAX_SAMPLES)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(f"Dataset size: {len(dataset)} positions")

    res = evaluate(model, loader, device)

    print("\n========== Inference Results ==========")
    print(f"Samples evaluated : {res['samples']}")
    print(f"Policy CE loss    : {res['ce']:.4f}")
    print(f"Policy top-1 acc  : {res['acc']*100:.2f}%")
    print("---------------------------------------")
    print(f"Delta MSE         : {res['mse']:.2f}")
    print(f"Delta RMSE (cp)   : {res['rmse']:.2f}")
    print(f"Delta MAE  (cp)   : {res['mae']:.2f}")
    print("=======================================\n")



if __name__ == "__main__":
    main()
