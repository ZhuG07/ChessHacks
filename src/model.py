import torch
import torch.nn as nn


# ============================================================
# BASIC RESIDUAL BLOCK (NO CBAM)
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
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)

        h = self.conv2(h)
        h = self.bn2(h)

        out = h + self.shortcut(x)
        out = self.relu(out)
        return out


# ============================================================
# RESNET20-LIKE TRUNK + POLICY + CP + DELTA + RESULT
# ============================================================

class ResNet20_PolicyDelta(nn.Module):
    """
    Simplified ResNet20-like model for ChessHacks:

      - Plain residual blocks (no CBAM)
      - Shared trunk for all heads
      - Heads:
          * policy_logits (policy_dim)
          * delta_cp (scaled & unscaled)
          * cp_before (scaled & unscaled)
          * game_result in [-1, 1] via tanh

    Forward returns:
      policy_logits, delta_real, delta_scaled, cp_real, cp_scaled, result

    Note:
      - cp_* heads are trained on a bounded, non-linear transform of centipawns,
        but cp_real = cp_scaled * cp_scale still provides a monotonic "CP-like"
        score you can use inside the engine.
    """

    def __init__(self, in_planes=18, policy_dim=20480, cp_scale=200.0, trunk_channels=128):
        super().__init__()

        self.cp_scale = cp_scale

        # ----------------------------------------------------
        # Stem
        # ----------------------------------------------------
        self.conv_in = nn.Conv2d(in_planes, trunk_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in   = nn.BatchNorm2d(trunk_channels)
        self.relu    = nn.ReLU(inplace=True)

        # ----------------------------------------------------
        # Trunk: 6 residual blocks at constant width
        # ----------------------------------------------------
        blocks = []
        for _ in range(6):
            blocks.append(BasicBlock(trunk_channels, trunk_channels))
        self.trunk = nn.Sequential(*blocks)

        # ----------------------------------------------------
        # Global average pooling
        # ----------------------------------------------------
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        # ----------------------------------------------------
        # Shared projection
        # ----------------------------------------------------
        self.fc_shared = nn.Linear(trunk_channels, 256)

        # Small gating MLP on shared features
        self.gate = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.Sigmoid(),
        )

        # ----------------------------------------------------
        # Heads
        # ----------------------------------------------------
        self.policy_head = nn.Linear(256, policy_dim)
        self.delta_head  = nn.Linear(256, 1)
        self.cp_head     = nn.Linear(256, 1)
        self.result_head = nn.Linear(256, 1)

    def forward(self, x):
        # Stem
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu(x)

        # Trunk
        x = self.trunk(x)

        # GAP
        x = self.avg(x).view(x.size(0), -1)

        # Shared projection
        x = self.fc_shared(x)

        # Gating
        gates = self.gate(x)
        x = self.relu(x * gates)

        # Policy
        policy_logits = self.policy_head(x)

        # Scalar heads (unscaled internal units)
        delta_scaled = self.delta_head(x).squeeze(-1)   # [B]
        cp_scaled    = self.cp_head(x).squeeze(-1)      # [B]
        result_raw   = self.result_head(x).squeeze(-1)  # [B]

        # "CP-like" scores (still useful even though training labels are bounded)
        delta_real = delta_scaled * self.cp_scale
        cp_real    = cp_scaled * self.cp_scale

        # Game result in [-1, 1]
        result = torch.tanh(result_raw)

        return policy_logits, delta_real, delta_scaled, cp_real, cp_scaled, result
