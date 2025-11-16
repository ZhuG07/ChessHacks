# training/selfplay/train_selfplay_from_npz.py

from pathlib import Path
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

from src.model import ResNet20_PolicyDelta


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
        in_planes=NUM_PLANES,
        policy_dim=POLICY_DIM,
        cp_scale=CP_SCALE,
    ).to(device)

    # Load starting weights (pull last best weights as init)
    if MODEL_PATH.is_file():
        print(f"[TRAIN] Loading initial weights from {MODEL_PATH}")
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state, strict=False)
    else:
        print(f"[TRAIN] WARNING: MODEL_PATH {MODEL_PATH} not found; training from scratch.")

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
