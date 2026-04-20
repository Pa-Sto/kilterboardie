from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    """
    Pick a GroupNorm group count that divides `channels`.
    Prefers larger groups for better normalization granularity.
    """
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Standard sinusoidal timestep embedding used in diffusion models.

    For frequency w_k, embedding uses:
      emb[2k]   = cos(t * w_k)
      emb[2k+1] = sin(t * w_k)
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device) / max(half, 1))
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    """
    Residual block with FiLM-like conditioning:
    - normalize + conv
    - add projected conditioning vector
    - normalize + conv
    - residual skip
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.norm2 = nn.GroupNorm(_group_count(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.cond_proj(cond)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class KilterDiffusionUNet(nn.Module):
    """
    Conditional U-Net denoiser for route channels.

    Inputs:
      x_t   : noisy route tensor at diffusion step t, shape (B, 4, H, W)
      static: board static context (hold mask/size/orientation), shape (B, S, H, W)
      grade : discrete grade index, shape (B,)
      t     : diffusion timestep index, shape (B,)

    Output:
      eps_hat: predicted Gaussian noise, shape (B, 4, H, W)
    """

    def __init__(
        self,
        num_grades: int,
        static_channels: int,
        route_channels: int = 4,
        base_channels: int = 64,
        grade_emb_dim: int = 32,
        time_emb_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_grades = num_grades
        self.static_channels = static_channels
        self.route_channels = route_channels
        self.base_channels = base_channels
        self.grade_emb_dim = grade_emb_dim
        self.time_emb_dim = time_emb_dim

        self.grade_emb = nn.Embedding(num_grades, grade_emb_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(time_emb_dim + grade_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        in_ch = route_channels + static_channels
        self.in_conv = nn.Conv2d(in_ch, base_channels, kernel_size=3, padding=1)

        self.down1 = ResBlock(base_channels, base_channels, time_emb_dim)
        self.downsample1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.down2 = ResBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)

        self.mid1 = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.mid2 = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        self.up2_reduce = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)
        self.up2 = ResBlock(base_channels * 4, base_channels * 2, time_emb_dim)
        self.up1_reduce = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.up1 = ResBlock(base_channels * 2, base_channels, time_emb_dim)

        self.out_norm = nn.GroupNorm(_group_count(base_channels), base_channels)
        self.out_conv = nn.Conv2d(base_channels, route_channels, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, static: torch.Tensor, grade: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Build conditioning vector from timestep and grade.
        t_emb = timestep_embedding(t, self.time_emb_dim)
        g_emb = self.grade_emb(grade)
        cond = self.cond_mlp(torch.cat([t_emb, g_emb], dim=1))

        # Concatenate noisy dynamic channels with static board context.
        x = torch.cat([x_t, static], dim=1)
        x0 = self.in_conv(x)

        # Encoder path (global context).
        d1 = self.down1(x0, cond)
        d2 = self.down2(self.downsample1(d1), cond)

        # Bottleneck.
        m = self.downsample2(d2)
        m = self.mid1(m, cond)
        m = self.mid2(m, cond)

        # Decoder path with skip connections (recover precise hold locations).
        u2 = F.interpolate(m, size=d2.shape[2:], mode="nearest")
        u2 = self.up2_reduce(u2)
        u2 = self.up2(torch.cat([u2, d2], dim=1), cond)

        u1 = F.interpolate(u2, size=d1.shape[2:], mode="nearest")
        u1 = self.up1_reduce(u1)
        u1 = self.up1(torch.cat([u1, d1], dim=1), cond)

        out = self.out_conv(F.silu(self.out_norm(u1)))
        return out


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    Gather per-timestep scalar coefficients and reshape to broadcast over x.
    """
    out = a.to(t.device).gather(0, t)
    return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion:
    """
    DDPM forward/reverse process with linear beta schedule.

    Forward process:
      q(x_t | x_0) = N( sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I )
      x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

    Reverse model predicts eps_hat; x_0 is reconstructed by:
      x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_hat) / sqrt(alpha_bar_t)
    """

    def __init__(self, timesteps: int = 200, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        self.timesteps = int(timesteps)

        # beta_t linearly increases noise over time.
        betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(posterior_variance.clamp_min(1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample x_t from q(x_t | x_0):
          x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        return _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + _extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        ) * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """
        Recover x_0 estimate from x_t and eps_hat:
          x0_hat = x_t / sqrt(alpha_bar_t) - eps_hat * sqrt(1/alpha_bar_t - 1)
        """
        return _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - _extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        ) * eps

    def p_mean_variance(
        self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior parameters p_theta(x_{t-1} | x_t):
        - convert eps_hat -> x0_hat
        - plug x0_hat into closed-form q(x_{t-1} | x_t, x_0) mean/variance
        """
        x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred).clamp(-1.0, 1.0)
        mean = _extract(self.posterior_mean_coef1, t, x_t.shape) * x0_pred + _extract(
            self.posterior_mean_coef2, t, x_t.shape
        ) * x_t
        var = _extract(self.posterior_variance, t, x_t.shape)
        log_var = _extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var, x0_pred

    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        static: torch.Tensor,
        grade: torch.Tensor,
        hold_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One reverse step:
          x_{t-1} = mean_theta(x_t, t) + sigma_t * z

        If hold_mask is given, invalid board cells are forced to -1
        in latent space so route channels remain restricted to valid holds.
        """
        eps_pred = model(x_t, static, grade, t)
        mean, _, log_var, x0_pred = self.p_mean_variance(x_t, t, eps_pred)
        noise = torch.randn_like(x_t)
        nonzero = (t > 0).float().view(-1, 1, 1, 1)
        x_prev = mean + nonzero * torch.exp(0.5 * log_var) * noise

        if hold_mask is not None:
            if hold_mask.shape[1] == 1:
                hold_mask = hold_mask.expand_as(x_prev)
            x_prev = x_prev * hold_mask + (-1.0) * (1.0 - hold_mask)
            x0_pred = x0_pred * hold_mask + (-1.0) * (1.0 - hold_mask)

        return x_prev, x0_pred

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, int, int, int],
        static: torch.Tensor,
        grade: torch.Tensor,
        device: torch.device,
        hold_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full ancestral sampling loop from t=T-1 ... 0.
        Returns x0_hat in [-1, 1].
        """
        x = torch.randn(shape, device=device)
        if hold_mask is not None:
            if hold_mask.shape[1] == 1:
                hold_mask = hold_mask.expand_as(x)
            x = x * hold_mask + (-1.0) * (1.0 - hold_mask)

        x0_pred = None
        for step in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            x, x0_pred = self.p_sample(model, x, t, static=static, grade=grade, hold_mask=hold_mask)

        if x0_pred is None:
            x0_pred = x
        return x0_pred.clamp(-1.0, 1.0)


def masked_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    hold_mask: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    targets_binary: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Weighted MSE over valid hold cells.

    Used for diffusion denoising loss L_eps = ||eps - eps_hat||^2, with
    optional upweighting where route target is positive to address sparsity.
    """
    mask = hold_mask
    if mask.shape[1] == 1:
        mask = mask.expand_as(pred)
    weight = torch.ones_like(pred)
    if pos_weight is not None and targets_binary is not None:
        pw = pos_weight.to(pred.device).view(1, -1, 1, 1)
        weight = torch.where(targets_binary > 0.5, pw, weight)
    total_weight = (mask * weight).sum().clamp_min(1.0)
    return (((pred - target) ** 2) * mask * weight).sum() / total_weight


def masked_weighted_bce_prob(
    probs: torch.Tensor,
    targets: torch.Tensor,
    hold_mask: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    BCE on probabilities (not logits), masked to valid holds.
    Optional positive-class weighting per channel.
    """
    probs = probs.clamp(1e-5, 1.0 - 1e-5)
    bce = F.binary_cross_entropy(probs, targets, reduction="none")
    if pos_weight is not None:
        pw = pos_weight.to(probs.device).view(1, -1, 1, 1)
        bce = torch.where(targets > 0.5, bce * pw, bce)
    mask = hold_mask
    if mask.shape[1] == 1:
        mask = mask.expand_as(bce)
    denom = mask.sum().clamp_min(1.0)
    return (bce * mask).sum() / denom


def structure_losses(
    probs: torch.Tensor,
    targets: torch.Tensor,
    hold_mask: torch.Tensor,
    count_min: int = 1,
    count_max: int = 2,
    path_reach: int = 10,
    path_steps: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiable structural regularizers reused from CVAE design:
    - count_loss : match start/finish counts to the target sample counts
    - path_loss  : encourage finish nodes to be reachable from starts
    - upward_loss: penalize hand probability mass below starts / above finishes
    - hand_density_loss: keep hand occupancy near the real route count
    - foot_density_loss: keep foot occupancy near the real route count,
      with extra penalty for overprediction
    """
    mask = hold_mask
    if mask.shape[1] == 1:
        mask = mask.expand_as(probs[:, 0:1])

    pred_start = (probs[:, 0:1] * mask).sum(dim=(1, 2, 3))
    pred_finish = (probs[:, 1:2] * mask).sum(dim=(1, 2, 3))
    true_start = (targets[:, 0:1] * mask).sum(dim=(1, 2, 3))
    true_finish = (targets[:, 1:2] * mask).sum(dim=(1, 2, 3))

    target_start = true_start.clamp(min=count_min, max=count_max)
    target_finish = true_finish.clamp(min=count_min, max=count_max)
    start_count_loss = (pred_start - target_start).abs().mean()
    finish_count_loss = (pred_finish - target_finish).abs().mean()
    count_loss = start_count_loss + 1.5 * finish_count_loss

    start = probs[:, 0:1] * mask
    finish = probs[:, 1:2] * mask
    hand = probs[:, 2:3] * mask
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

    b, _, h, w = hand.shape
    rows = torch.arange(h, device=probs.device, dtype=probs.dtype).view(1, 1, h, 1).expand(b, 1, h, w)

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

    pred_hand = (probs[:, 2:3] * mask).sum(dim=(1, 2, 3))
    pred_foot = (probs[:, 3:4] * mask).sum(dim=(1, 2, 3))
    true_hand = (targets[:, 2:3] * mask).sum(dim=(1, 2, 3))
    true_foot = (targets[:, 3:4] * mask).sum(dim=(1, 2, 3))

    hand_over = F.relu(pred_hand - true_hand)
    hand_under = F.relu(true_hand - pred_hand)
    foot_over = F.relu(pred_foot - true_foot)
    foot_under = F.relu(true_foot - pred_foot)
    hand_density_loss = (2.0 * hand_over + 0.5 * hand_under).mean()
    foot_density_loss = (3.0 * foot_over + 0.5 * foot_under).mean()

    return count_loss, path_loss, upward_loss, hand_density_loss, foot_density_loss
