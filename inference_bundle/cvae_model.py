from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KilterCVAE(nn.Module):
    """
    Conditional VAE for Kilterboard routes.

    Inputs:
      - route: (B, 4, H, W)
      - static: (B, 2, H, W) [hold_presence, hold_size]
      - grade: (B,) int64 in [0, num_grades-1]

    Output:
      - logits for 4 route channels (B, 4, H, W)
    """

    def __init__(
        self,
        num_grades: int,
        emb_dim: int = 16,
        latent_dim: int = 64,
        static_channels: int = 2,
        in_h: int = 34,
        in_w: int = 35,
    ) -> None:
        super().__init__()
        self.num_grades = num_grades
        self.emb_dim = emb_dim
        self.latent_dim = latent_dim
        self.static_channels = static_channels
        self.in_h = in_h
        self.in_w = in_w

        self.grade_emb = nn.Embedding(num_grades, emb_dim)

        enc_in = 4 + static_channels + emb_dim
        self.enc_conv = nn.Sequential(
            nn.Conv2d(enc_in, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, enc_in, in_h, in_w)
            dummy_out = self.enc_conv(dummy)
            self._enc_h = dummy_out.shape[2]
            self._enc_w = dummy_out.shape[3]
            self._enc_flat = dummy_out.numel()

        self.enc_mu = nn.Linear(self._enc_flat, latent_dim)
        self.enc_logvar = nn.Linear(self._enc_flat, latent_dim)

        self.dec_fc = nn.Linear(latent_dim + emb_dim, 128 * self._enc_h * self._enc_w)
        self.dec_deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )

        self.dec_out = nn.Sequential(
            nn.Conv2d(32 + static_channels, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 4, kernel_size=1),
        )

    def _grade_map(self, grade: torch.Tensor, h: int, w: int) -> torch.Tensor:
        emb = self.grade_emb(grade)  # (B, emb_dim)
        emb = emb[:, :, None, None].expand(-1, -1, h, w)
        return emb

    def encode(self, route: torch.Tensor, static: torch.Tensor, grade: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grade_map = self._grade_map(grade, route.shape[2], route.shape[3])
        x = torch.cat([route, static, grade_map], dim=1)
        h = self.enc_conv(x)
        h = h.view(h.shape[0], -1)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, static: torch.Tensor, grade: torch.Tensor) -> torch.Tensor:
        grade_emb = self.grade_emb(grade)
        z = torch.cat([z, grade_emb], dim=1)
        h = self.dec_fc(z)
        h = h.view(h.shape[0], 128, self._enc_h, self._enc_w)
        h = self.dec_deconv(h)
        if h.shape[2] != static.shape[2] or h.shape[3] != static.shape[3]:
            raise RuntimeError(f"Decoder output shape {h.shape[2:]} does not match static shape {static.shape[2:]}")
        h = torch.cat([h, static], dim=1)
        logits = self.dec_out(h)
        return logits

    def forward(self, route: torch.Tensor, static: torch.Tensor, grade: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(route, static, grade)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, static, grade)
        return logits, mu, logvar

    def sample(self, grade: torch.Tensor, static: torch.Tensor, n: int = 1) -> torch.Tensor:
        if grade.ndim == 0:
            grade = grade.unsqueeze(0)
        if grade.shape[0] != n:
            grade = grade.expand(n)
        z = torch.randn(n, self.latent_dim, device=grade.device)
        return self.decode(z, static, grade)


def cvae_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    hold_mask: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.01,
    pos_weight: torch.Tensor = None,
    count_weight: float = 0.0,
    count_min: int = 1,
    count_max: int = 2,
    focal_gamma: float = 0.0,
    path_weight: float = 0.0,
    path_reach: int = 10,
    path_steps: int = 4,
    upward_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (total_loss, recon_loss, kl_loss)
    """
    if pos_weight is not None:
        if pos_weight.ndim == 1:
            pos_weight = pos_weight.view(1, -1, 1, 1)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction='none')
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    if focal_gamma and focal_gamma > 0.0:
        pt = torch.exp(-bce)
        bce = ((1.0 - pt) ** focal_gamma) * bce

    mask = hold_mask
    if mask.shape[1] == 1:
        mask = mask.expand_as(bce)
    bce = bce * mask
    denom = mask.sum().clamp_min(1.0)
    recon = bce.sum() / denom

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    prob = None
    if count_weight > 0.0 or path_weight > 0.0 or upward_weight > 0.0:
        prob = torch.sigmoid(logits)

    count_loss = torch.tensor(0.0, device=logits.device)
    if count_weight > 0.0:
        # Predicted counts (expected number of holds) for start and finish
        mask = hold_mask
        if mask.shape[1] == 1:
            mask = mask.expand_as(prob[:, 0:1])

        pred_start = (prob[:, 0:1] * mask).sum(dim=(1, 2, 3))
        pred_finish = (prob[:, 1:2] * mask).sum(dim=(1, 2, 3))
        true_start = (targets[:, 0:1] * mask).sum(dim=(1, 2, 3))
        true_finish = (targets[:, 1:2] * mask).sum(dim=(1, 2, 3))

        target_start = true_start.clamp(min=count_min, max=count_max)
        target_finish = true_finish.clamp(min=count_min, max=count_max)

        count_loss = (pred_start - target_start).abs().mean() + (pred_finish - target_finish).abs().mean()

    path_loss = torch.tensor(0.0, device=logits.device)
    if path_weight > 0.0:
        mask = hold_mask
        if mask.shape[1] == 1:
            mask = mask.expand_as(prob[:, 0:1])
        start = prob[:, 0:1] * mask
        finish = prob[:, 1:2] * mask
        hand = prob[:, 2:3] * mask
        nodes = torch.clamp(start + finish + hand, 0.0, 1.0)

        reachable = start * nodes
        reach = int(path_reach)
        steps = int(path_steps)
        if reach > 0 and steps > 0:
            kernel = 2 * reach + 1
            for _ in range(steps):
                reachable = F.max_pool2d(reachable, kernel_size=kernel, stride=1, padding=reach)
                reachable = reachable * nodes

        finish_sum = finish.sum(dim=(1, 2, 3)).clamp_min(1e-6)
        reachable_finish = (reachable * finish).sum(dim=(1, 2, 3))
        score = reachable_finish / finish_sum
        path_loss = (1.0 - score).mean()

    upward_loss = torch.tensor(0.0, device=logits.device)
    if upward_weight > 0.0:
        mask = hold_mask
        if mask.shape[1] == 1:
            mask = mask.expand_as(prob[:, 0:1])
        start = prob[:, 0:1] * mask
        finish = prob[:, 1:2] * mask
        hand = prob[:, 2:3] * mask

        b, _, h, w = hand.shape
        rows = torch.arange(h, device=logits.device, dtype=logits.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        start_sum = start.sum(dim=(1, 2, 3)).clamp_min(1e-6)
        finish_sum = finish.sum(dim=(1, 2, 3)).clamp_min(1e-6)
        start_avg = (start * rows).sum(dim=(1, 2, 3)) / start_sum
        finish_avg = (finish * rows).sum(dim=(1, 2, 3)) / finish_sum

        start_avg = start_avg.view(b, 1, 1, 1)
        finish_avg = finish_avg.view(b, 1, 1, 1)

        below_start = F.relu(rows - start_avg)
        above_finish = F.relu(finish_avg - rows)
        penalty = (below_start + above_finish) / max(h - 1, 1)

        hand_sum = hand.sum(dim=(1, 2, 3)).clamp_min(1e-6)
        upward_loss = (hand * penalty).sum(dim=(1, 2, 3)) / hand_sum
        upward_loss = upward_loss.mean()

    total = recon + beta * kl + count_weight * count_loss + path_weight * path_loss + upward_weight * upward_loss
    return total, recon, kl, count_loss, path_loss, upward_loss
