"""`ave.py` 의미를 유지하는 residual TADGAN 모델 정의."""

import os
import tomllib

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .utils import (
    flatten_model_config_dict,
    normalize_bounded_head_config,
    summarize_model_config,
)


VERSION_MAP = {"TADGAN": "ave"}


class TADGANConfig:
    """`ave.py` 기본 철학에 residual skeleton을 얹은 설정."""

    def __init__(self):
        self.epochs = 1000
        self.latent_dim = 16
        self.noise_dim = 16
        self.timesteps = 1000
        self.lr = 1e-3
        self.batch_size = 256
        self.num_workers = 0
        self.bin_threshold = 0.5
        self.sampling_strategy = "prior"
        self.w_rec = 1.0
        self.w_kl = 0.01
        self.w_cls = 1.0
        self.alpha = 0.5

        self.use_lr_scheduler = False
        self.use_r1_penalty = False
        self.r1_weight = 10.0
        self.batch_sampling_strategy = "natural"
        self.minority_quota_trigger_ratio = 0.03
        self.minority_quota_trigger_expected = 8.0
        self.minority_quota_trigger_zero_prob = 0.02
        self.minority_max_ratio = 0.05
        self.use_generator_ema = False
        self.use_mixed_precision = False
        self.grad_clip_norm = 1.0
        self.use_label_smoothing = False
        self.label_smoothing = 0.05
        self.use_wide_condition_embedding = False
        self.use_mode_seeking_regularization = True
        self.mode_seeking_weight = 0.01
        self.mode_seeking_eps = 1e-6
        self.latent_align_weight = 0.01
        self.stage1_end_epoch = None
        self.stage2_end_epoch = None
        self.stage1_ratio = 0.2
        self.stage2_ratio = 0.4
        self.stage3_ratio = 0.4
        self.stage3_mode_seeking_scale = 0.5
        self.lr_scheduler_type = "cosine"
        self.lr_scheduler_t_max = None
        self.use_bounded_head = False
        self.use_continuous_clip = False
        self.use_residual_encoder = True
        self.use_residual_decoder = True
        self.use_spectral_norm_discriminator = False
        self.use_two_layer_decoder_heads = True

        self.encoder_hidden_dim = 128
        self.decoder_hidden_dim = 128
        self.decoder_shared_dim = 64
        self.decoder_head_hidden_dim = 64
        self.generator_hidden_dim = 128
        self.discriminator_hidden_dim = 128
        self.encoder_num_blocks = 2
        self.decoder_num_blocks = 2
        self.generator_num_blocks = 2
        self.discriminator_num_blocks = 2

    def to_dict(self):
        return dict(self.__dict__)

    def load_config(self, config_path, verbose=True):
        if not os.path.exists(config_path):
            if verbose:
                tqdm.write(f"[WARN] config.toml not found: {config_path} | default config 사용")
            return

        with open(config_path, "rb") as file:
            config = normalize_bounded_head_config(flatten_model_config_dict(tomllib.load(file)))

        valid_keys = set(self.__dict__.keys())
        ignored_keys = []
        for key, value in config.items():
            if key in valid_keys:
                setattr(self, key, value)
            else:
                ignored_keys.append(key)
        if verbose:
            tqdm.write(summarize_model_config("TADGAN", self))
            if ignored_keys:
                tqdm.write(f"[WARN] ignored config keys: {', '.join(sorted(ignored_keys))}")


def _make_linear(in_dim, out_dim, use_spectral_norm=False):
    layer = nn.Linear(in_dim, out_dim)
    if use_spectral_norm:
        layer = nn.utils.spectral_norm(layer)
    return layer


class ResidualMLPBlock(nn.Module):
    """LayerNorm 기반 residual MLP block."""

    def __init__(self, hidden_dim, use_spectral_norm=False):
        super().__init__()
        self.fc1 = _make_linear(hidden_dim, hidden_dim, use_spectral_norm=use_spectral_norm)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = _make_linear(hidden_dim, hidden_dim, use_spectral_norm=use_spectral_norm)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.fc2(out)
        out = self.norm2(out)
        return self.act(out + x)


class MLPProjector(nn.Module):
    """Linear -> LayerNorm -> ReLU projector."""

    def __init__(self, in_dim, out_dim, use_spectral_norm=False):
        super().__init__()
        self.fc = _make_linear(in_dim, out_dim, use_spectral_norm=use_spectral_norm)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.norm(x)
        return self.act(x)


class Encoder(nn.Module):
    """Input feature를 latent z0로 압축."""

    def __init__(self, input_dim, latent_dim, hidden_dim=128, num_blocks=2, use_residual=True):
        super().__init__()
        self.use_residual = use_residual

        if use_residual:
            self.input_proj = MLPProjector(input_dim, hidden_dim)
            self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim) for _ in range(num_blocks)])
            self.mid_proj = MLPProjector(hidden_dim, 64)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
            )

        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        if self.use_residual:
            h = self.input_proj(x)
            for block in self.blocks:
                h = block(h)
            h = self.mid_proj(h)
        else:
            h = self.net(x)

        mu = self.mu(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class Diffusion(nn.Module):
    """Forward noising만 담당하는 diffusion module."""

    def __init__(self, timesteps, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cum = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps
        self.register_buffer("sqrt_ac", alphas_cum.sqrt().float())
        self.register_buffer("sqrt_1m_ac", (1.0 - alphas_cum).sqrt().float())

    def q_sample(self, z0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        a = self.sqrt_ac[t].unsqueeze(1)
        b = self.sqrt_1m_ac[t].unsqueeze(1)
        return a * z0 + b * noise

    def forward(self, z0):
        batch = z0.size(0)
        t = torch.randint(0, self.timesteps, (batch,), device=z0.device)
        noise = torch.randn_like(z0)
        zt = self.q_sample(z0, t, noise=noise)
        return zt, t


class Generator(nn.Module):
    """`ave.py` 의미를 따르는 조건부 latent generator."""

    def __init__(self, noise_dim, latent_dim, num_classes, timesteps,
                 hidden_dim=128, num_blocks=2, t_emb_dim=16, y_emb_dim=4):
        super().__init__()
        self.t_emb = nn.Embedding(timesteps, t_emb_dim)
        self.y_emb = nn.Embedding(num_classes, y_emb_dim)
        in_dim = noise_dim + t_emb_dim + y_emb_dim

        self.input_proj = MLPProjector(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim) for _ in range(num_blocks)])
        self.mid_proj = MLPProjector(hidden_dim, 64)
        self.head = nn.Linear(64, latent_dim)
        self.noise_dim = noise_dim

    def forward(self, noise, t, y):
        te = self.t_emb(t)
        ye = self.y_emb(y)
        h = torch.cat([noise, te, ye], dim=1)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h)
        h = self.mid_proj(h)
        return self.head(h)


class Discriminator(nn.Module):
    """`ave.py` 의미를 따르는 AC-GAN 형태 discriminator."""

    def __init__(self, latent_dim, num_classes, timesteps,
                 hidden_dim=128, num_blocks=2, t_emb_dim=16, y_emb_dim=4,
                 use_spectral_norm=False):
        super().__init__()
        self.t_emb = nn.Embedding(timesteps, t_emb_dim)
        self.y_emb = nn.Embedding(num_classes, y_emb_dim)
        in_dim = latent_dim + t_emb_dim + y_emb_dim

        self.input_proj = MLPProjector(in_dim, hidden_dim, use_spectral_norm=use_spectral_norm)
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, use_spectral_norm=use_spectral_norm)
            for _ in range(num_blocks)
        ])
        self.mid_proj = MLPProjector(hidden_dim, 64, use_spectral_norm=use_spectral_norm)
        self.adv_head = _make_linear(64, 1, use_spectral_norm=use_spectral_norm)
        self.cls_head = _make_linear(64, num_classes, use_spectral_norm=use_spectral_norm)

    def forward(self, z, t, y):
        te = self.t_emb(t)
        ye = self.y_emb(y)
        h = torch.cat([z, te, ye], dim=1)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h)
        h = self.mid_proj(h)
        return self.adv_head(h), self.cls_head(h)


class DecoderMixed(nn.Module):
    """연속형과 이산형 feature를 동시에 복원하는 decoder."""

    def __init__(self, latent_dim, con_dim, bin_dim, hidden_dim=128,
                 num_blocks=2, use_residual=True, use_two_layer_heads=True,
                 shared_dim=64, head_hidden_dim=64, use_bounded_head=False):
        super().__init__()
        self.use_residual = use_residual
        self.use_two_layer_heads = use_two_layer_heads
        self.con_dim = con_dim
        self.use_bounded_head = bool(use_bounded_head and con_dim > 0)
        self.register_buffer("con_min_scaled", torch.zeros(con_dim, dtype=torch.float32), persistent=True)
        self.register_buffer("con_max_scaled", torch.zeros(con_dim, dtype=torch.float32), persistent=True)
        self.register_buffer("has_continuous_bounds", torch.tensor(False, dtype=torch.bool), persistent=True)

        if use_residual:
            self.input_proj = MLPProjector(latent_dim, hidden_dim)
            self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim) for _ in range(num_blocks)])
            self.shared_proj = MLPProjector(hidden_dim, shared_dim)
        else:
            self.shared = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, shared_dim),
                nn.ReLU(),
            )

        if con_dim > 0:
            if use_two_layer_heads:
                self.hc = nn.Sequential(
                    nn.Linear(shared_dim, head_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(head_hidden_dim, con_dim),
                )
            else:
                self.hc = nn.Linear(shared_dim, con_dim)
        else:
            self.hc = None

        if bin_dim > 0:
            if use_two_layer_heads:
                self.hb = nn.Sequential(
                    nn.Linear(shared_dim, head_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(head_hidden_dim, bin_dim),
                )
            else:
                self.hb = nn.Linear(shared_dim, bin_dim)
        else:
            self.hb = None

    def set_continuous_bounds(self, con_min_scaled, con_max_scaled):
        if self.con_dim <= 0 or con_min_scaled is None or con_max_scaled is None:
            self.has_continuous_bounds.fill_(False)
            return

        low = torch.as_tensor(con_min_scaled, dtype=self.con_min_scaled.dtype, device=self.con_min_scaled.device).flatten()
        high = torch.as_tensor(con_max_scaled, dtype=self.con_max_scaled.dtype, device=self.con_max_scaled.device).flatten()
        if low.numel() != self.con_dim or high.numel() != self.con_dim:
            self.has_continuous_bounds.fill_(False)
            return

        self.con_min_scaled.copy_(low)
        self.con_max_scaled.copy_(high)
        self.has_continuous_bounds.fill_(True)

    def _apply_bounded_head(self, raw_con):
        if not self.use_bounded_head or not bool(self.has_continuous_bounds.item()):
            return raw_con, False

        low = self.con_min_scaled.to(device=raw_con.device, dtype=raw_con.dtype)
        high = self.con_max_scaled.to(device=raw_con.device, dtype=raw_con.dtype)
        bounded = low + 0.5 * (torch.tanh(raw_con) + 1.0) * (high - low)
        return bounded, True

    def forward(self, z):
        if self.use_residual:
            h = self.input_proj(z)
            for block in self.blocks:
                h = block(h)
            h = self.shared_proj(h)
        else:
            h = self.shared(z)

        out = {}
        if self.hc is not None:
            raw_con = self.hc(h)
            con_pred, is_bounded = self._apply_bounded_head(raw_con)
            out["x_hat_con"] = con_pred
            out["x_hat_cont"] = con_pred
            out["x_hat_con_raw"] = raw_con
            out["x_hat_con_bounded"] = is_bounded
        if self.hb is not None:
            out["x_hat_bin_logit"] = self.hb(h)
        return out


class TADGAN(nn.Module):
    """`ave.py` 의미를 유지하는 residual TADGAN."""

    def __init__(self, con_dim, bin_dim, config, num_classes):
        super().__init__()
        in_dim = con_dim + bin_dim
        cond_t_dim = 32 if config.use_wide_condition_embedding else 16
        cond_y_dim = 16 if config.use_wide_condition_embedding else 4

        self.con_dim = con_dim
        self.bin_dim = bin_dim
        self.noise_dim = config.noise_dim
        self.timesteps = config.timesteps

        self.encoder = Encoder(
            in_dim, config.latent_dim,
            hidden_dim=config.encoder_hidden_dim,
            num_blocks=config.encoder_num_blocks,
            use_residual=config.use_residual_encoder,
        )
        self.diffusion = Diffusion(config.timesteps)
        self.generator = Generator(
            config.noise_dim, config.latent_dim, num_classes, config.timesteps,
            hidden_dim=config.generator_hidden_dim,
            num_blocks=config.generator_num_blocks,
            t_emb_dim=cond_t_dim,
            y_emb_dim=cond_y_dim,
        )
        self.discriminator = Discriminator(
            config.latent_dim, num_classes, config.timesteps,
            hidden_dim=config.discriminator_hidden_dim,
            num_blocks=config.discriminator_num_blocks,
            t_emb_dim=cond_t_dim,
            y_emb_dim=cond_y_dim,
            use_spectral_norm=config.use_spectral_norm_discriminator,
        )
        self.decoder = DecoderMixed(
            config.latent_dim, con_dim, bin_dim,
            hidden_dim=config.decoder_hidden_dim,
            num_blocks=config.decoder_num_blocks,
            use_residual=config.use_residual_decoder,
            use_two_layer_heads=config.use_two_layer_decoder_heads,
            shared_dim=config.decoder_shared_dim,
            head_hidden_dim=config.decoder_head_hidden_dim,
            use_bounded_head=config.use_bounded_head,
        )

    def sample_noise(self, batch, device):
        return torch.randn(batch, self.noise_dim, device=device)

    def set_continuous_bounds(self, con_min_scaled, con_max_scaled):
        self.decoder.set_continuous_bounds(con_min_scaled, con_max_scaled)


def kl_loss(mu, logvar):
    return 0.5 * torch.mean(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)
