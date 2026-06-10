"""VAE 제거 baseline 모델."""

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
from generation.VEDP_GAN.model import Diffusion, MLPProjector, ResidualMLPBlock, _make_linear


class VAEFreeGANConfig:
    def __init__(self):
        self.epochs = 1000
        self.noise_dim = 16
        self.timesteps = 1000
        self.lr = 1e-3
        self.w_cls = 1.0
        self.alpha = 0.5
        self.batch_size = 256
        self.num_workers = 0
        self.bin_threshold = 0.5
        self.sampling_strategy = "prior"
        self.use_label_smoothing = False
        self.label_smoothing = 0.05
        self.use_lr_scheduler = False
        self.use_r1_penalty = False
        self.r1_weight = 10.0
        self.use_mixed_precision = False
        self.grad_clip_norm = 1.0
        self.batch_sampling_strategy = "natural"
        self.minority_quota_trigger_ratio = 0.03
        self.minority_quota_trigger_expected = 8.0
        self.minority_quota_trigger_zero_prob = 0.02
        self.minority_max_ratio = 0.05
        self.use_generator_ema = False
        self.use_wide_condition_embedding = False
        self.use_bounded_head = False
        self.use_continuous_clip = False
        self.generator_hidden_dim = 128
        self.generator_shared_dim = 64
        self.generator_head_hidden_dim = 64
        self.discriminator_hidden_dim = 128
        self.generator_num_blocks = 2
        self.discriminator_num_blocks = 2
        self.use_spectral_norm_discriminator = False

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
            tqdm.write(summarize_model_config("VAEFreeGAN", self))
            if ignored_keys:
                tqdm.write(f"[WARN] ignored config keys: {', '.join(sorted(ignored_keys))}")


class FeatureGenerator(nn.Module):
    def __init__(self, noise_dim, con_dim, bin_dim, num_classes, timesteps,
                 hidden_dim=256, num_blocks=2, t_emb_dim=32, y_emb_dim=16,
                 shared_dim=64, head_hidden_dim=64):
        super().__init__()
        self.t_emb = nn.Embedding(timesteps, t_emb_dim)
        self.y_emb = nn.Embedding(num_classes, y_emb_dim)
        in_dim = noise_dim + t_emb_dim + y_emb_dim

        self.input_proj = MLPProjector(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim) for _ in range(num_blocks)])
        self.mid_proj = MLPProjector(hidden_dim, 128)
        self.shared_proj = MLPProjector(128, shared_dim)
        self.hc = nn.Sequential(nn.Linear(shared_dim, head_hidden_dim), nn.ReLU(), nn.Linear(head_hidden_dim, con_dim)) if con_dim > 0 else None
        self.hb = nn.Sequential(nn.Linear(shared_dim, head_hidden_dim), nn.ReLU(), nn.Linear(head_hidden_dim, bin_dim)) if bin_dim > 0 else None
        self.noise_dim = noise_dim

    def forward(self, noise, t, y):
        te = self.t_emb(t)
        ye = self.y_emb(y)
        h = torch.cat([noise, te, ye], dim=1)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h)
        h = self.mid_proj(h)
        h = self.shared_proj(h)

        out = {}
        if self.hc is not None:
            con_pred = self.hc(h)
            out["x_hat_con"] = con_pred
            out["x_hat_cont"] = con_pred
        if self.hb is not None:
            out["x_hat_bin_logit"] = self.hb(h)
        return out


class FeatureDiscriminator(nn.Module):
    def __init__(self, input_dim, num_classes, timesteps,
                 hidden_dim=256, num_blocks=2, t_emb_dim=32, y_emb_dim=16,
                 use_spectral_norm=True):
        super().__init__()
        self.t_emb = nn.Embedding(timesteps, t_emb_dim)
        self.y_emb = nn.Embedding(num_classes, y_emb_dim)
        in_dim = input_dim + t_emb_dim + y_emb_dim

        self.input_proj = MLPProjector(in_dim, hidden_dim, use_spectral_norm=use_spectral_norm)
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, use_spectral_norm=use_spectral_norm)
            for _ in range(num_blocks)])
        
        self.mid_proj = MLPProjector(hidden_dim, 128, use_spectral_norm=use_spectral_norm)
        self.out_proj = MLPProjector(128, 64, use_spectral_norm=use_spectral_norm)
        self.adv_head = _make_linear(64, 1, use_spectral_norm=use_spectral_norm)
        self.cls_head = _make_linear(64, num_classes, use_spectral_norm=use_spectral_norm)

    def forward(self, x, t, y):
        te = self.t_emb(t)
        ye = self.y_emb(y)
        h = torch.cat([x, te, ye], dim=1)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h)
        h = self.mid_proj(h)
        h = self.out_proj(h)
        return self.adv_head(h), self.cls_head(h)


class VAEFreeGAN(nn.Module):
    def __init__(self, con_dim, bin_dim, config, num_classes):
        super().__init__()
        input_dim = con_dim + bin_dim
        cond_t_dim = 32 if config.use_wide_condition_embedding else 16
        cond_y_dim = 16 if config.use_wide_condition_embedding else 4
        self.con_dim = con_dim
        self.bin_dim = bin_dim
        self.noise_dim = config.noise_dim
        self.timesteps = config.timesteps
        self.diffusion = Diffusion(config.timesteps)
        self.generator = FeatureGenerator(
            config.noise_dim, con_dim, bin_dim, num_classes, config.timesteps,
            hidden_dim=config.generator_hidden_dim,
            num_blocks=config.generator_num_blocks,
            t_emb_dim=cond_t_dim,
            y_emb_dim=cond_y_dim,
            shared_dim=config.generator_shared_dim,
            head_hidden_dim=config.generator_head_hidden_dim )
        
        self.discriminator = FeatureDiscriminator(
            input_dim, num_classes, config.timesteps,
            hidden_dim=config.discriminator_hidden_dim,
            num_blocks=config.discriminator_num_blocks,
            t_emb_dim=cond_t_dim,
            y_emb_dim=cond_y_dim,
            use_spectral_norm=config.use_spectral_norm_discriminator )

    def sample_noise(self, batch, device):
        return torch.randn(batch, self.noise_dim, device=device)
