"""Diffusion-only baseline model."""

import math
import os
import tomllib

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from ablation_study.scripts.utils import (
    flatten_model_config_dict,
    normalize_bounded_head_config,
    summarize_model_config,
)
from generation.VEDP_GAN.model import DecoderMixed, Encoder, MLPProjector, ResidualMLPBlock


class DiffusionOnlyConfig:
    def __init__(self):
        self.epochs = 1000
        self.latent_dim = 16
        self.timesteps = 1000
        self.lr = 1e-3
        self.w_rec = 1.0
        self.w_kl = 0.01
        self.w_diff = 1.0
        self.batch_size = 256
        self.num_workers = 0
        self.bin_threshold = 0.5
        self.sampling_strategy = "prior"
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
        self.use_bounded_head = False
        self.use_continuous_clip = False
        self.use_residual_decoder = True
        self.use_residual_encoder = True
        self.use_wide_condition_embedding = False
        self.encoder_hidden_dim = 128
        self.decoder_hidden_dim = 128
        self.decoder_shared_dim = 64
        self.decoder_head_hidden_dim = 64
        self.denoiser_hidden_dim = 128
        self.encoder_num_blocks = 2
        self.decoder_num_blocks = 2
        self.denoiser_num_blocks = 2

    def to_dict(self):
        return dict(self.__dict__)

    def load_config(self, config_path, verbose=True):
        if not os.path.exists(config_path):
            if verbose:
                tqdm.write(f"[WARN] config.toml not found: {config_path} | using default config")
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
            tqdm.write(summarize_model_config("DiffusionOnly", self))
            if ignored_keys:
                tqdm.write(f"[WARN] ignored config keys: {', '.join(sorted(ignored_keys))}")


class LatentDenoiser(nn.Module):
    def __init__(self, latent_dim, num_classes, timesteps,
                 hidden_dim=256, num_blocks=2, t_emb_dim=32, y_emb_dim=16):
        super().__init__()
        self.t_emb = nn.Embedding(timesteps, t_emb_dim)
        self.y_emb = nn.Embedding(num_classes, y_emb_dim)
        in_dim = latent_dim + t_emb_dim + y_emb_dim
        self.input_proj = MLPProjector(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim) for _ in range(num_blocks)])
        self.mid_proj = MLPProjector(hidden_dim, 128)
        self.out_proj = MLPProjector(128, 64)
        self.head = nn.Linear(64, latent_dim)

    def forward(self, zt, t, y):
        te = self.t_emb(t)
        ye = self.y_emb(y)
        h = torch.cat([zt, te, ye], dim=1)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h)
        h = self.mid_proj(h)
        h = self.out_proj(h)
        return self.head(h)


class DiffusionSchedule(nn.Module):
    def __init__(self, timesteps, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cum = torch.cumprod(alphas, dim=0)
        alphas_cum_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64), alphas_cum[:-1]], dim=0)

        self.timesteps = timesteps
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", alphas_cum.float())
        self.register_buffer("alphas_cumprod_prev", alphas_cum_prev.float())
        self.register_buffer("sqrt_alphas_cumprod", alphas_cum.sqrt().float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cum).sqrt().float())
        self.register_buffer("sqrt_recip_alphas", (1.0 / alphas).sqrt().float())
        posterior_variance = betas * (1.0 - alphas_cum_prev) / (1.0 - alphas_cum)
        self.register_buffer("posterior_variance", posterior_variance.float())

    def q_sample(self, z0, t, noise):
        a = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        b = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        return a * z0 + b * noise

    def predict_x0(self, zt, t, noise_pred):
        a = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        b = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        return (zt - b * noise_pred) / torch.clamp(a, min=1e-8)

    def p_sample(self, zt, t, noise_pred):
        betas_t = self.betas[t].unsqueeze(1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        sqrt_recip_alpha = self.sqrt_recip_alphas[t].unsqueeze(1)
        model_mean = sqrt_recip_alpha * (zt - betas_t * noise_pred / torch.clamp(sqrt_one_minus, min=1e-8))

        nonzero_mask = (t != 0).float().unsqueeze(1)
        posterior_var = self.posterior_variance[t].unsqueeze(1)
        noise = torch.randn_like(zt)
        return model_mean + nonzero_mask * torch.sqrt(torch.clamp(posterior_var, min=1e-12)) * noise


class DiffusionOnlyModel(nn.Module):
    def __init__(self, con_dim, bin_dim, config, num_classes):
        super().__init__()
        in_dim = con_dim + bin_dim
        cond_t_dim = 32 if config.use_wide_condition_embedding else 16
        cond_y_dim = 16 if config.use_wide_condition_embedding else 4
        self.con_dim = con_dim
        self.bin_dim = bin_dim
        self.timesteps = config.timesteps
        self.encoder = Encoder(
            in_dim, config.latent_dim,
            hidden_dim=config.encoder_hidden_dim,
            num_blocks=config.encoder_num_blocks,
            use_residual=config.use_residual_encoder,
        )
        self.decoder = DecoderMixed(
            config.latent_dim, con_dim, bin_dim,
            hidden_dim=config.decoder_hidden_dim,
            num_blocks=config.decoder_num_blocks,
            use_residual=config.use_residual_decoder,
            use_two_layer_heads=True,
            shared_dim=config.decoder_shared_dim,
            head_hidden_dim=config.decoder_head_hidden_dim,
        )
        self.schedule = DiffusionSchedule(config.timesteps)
        self.denoiser = LatentDenoiser(
            config.latent_dim, num_classes, config.timesteps,
            hidden_dim=config.denoiser_hidden_dim,
            num_blocks=config.denoiser_num_blocks,
            t_emb_dim=cond_t_dim,
            y_emb_dim=cond_y_dim,
        )
