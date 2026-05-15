"""TTGAN 실험 설정을 TOML에서 불러오기 위한 헬퍼"""

from __future__ import annotations

import os
import tomllib


class TTGANConfig:
    """TTGAN 학습/샘플링 설정을 관리함"""

    def __init__(self):
        # 기본 학습 파라미터
        self.epochs = 2000
        self.embedding_dim = 128
        self.generator_num_layers = 6
        self.discriminator_dim = [256, 256]
        self.generator_lr = 2e-4
        self.generator_decay = 1e-6
        self.discriminator_lr = 2e-4
        self.discriminator_decay = 1e-6
        self.batch_size = 1000
        self.discriminator_steps = 1
        self.log_frequency = True
        self.pac = 10
        self.cuda = True

        # 고급 학습 옵션
        self.gradient_penalty_lambda = 10.0
        self.cond_loss_weight = 1.0
        self.use_dynamic_weights = False
        self.dynamic_weight_start = 0.5
        self.dynamic_weight_end = 1.0
        self.use_kl_anneal = False
        self.use_lr_scheduler = False
        self.use_r1_penalty = False
        self.r1_weight = 10.0
        self.use_generator_ema = False
        self.ema_decay = 0.999
        self.use_mixed_precision = False
        self.grad_clip_norm = 1.0
        self.use_label_smoothing = False
        self.label_smoothing = 0.05
        self.classwise_training = False

        # 샘플링 옵션
        self.enable_sampling_noise = True
        self.sampling_noise_std_ratio = 0.01
        self.use_discretized_rounding_logic = True

        # 기타
        self.use_residual_discriminator = False
        self.discriminator_residual_layers = 0
        self.discriminator_residual_dropout = 0.3

    def _toml_path(self, args) -> str:
        return os.path.join(args.exp_dir, 'TTGAN', 'config.toml')

    def load_from_exp(self, args, verbose=True):
        """exp 디렉터리의 config.toml을 읽어 설정을 갱신함"""
        path = self._toml_path(args)
        if not os.path.exists(path):
            if verbose:
                print(f"⚠️ {path}가 존재하지 않아 기본 설정으로 학습을 진행합니다.")
            return
        with open(path, 'rb') as fp:
            if verbose:
                print(f"✅️ {path}에서 설정을 불러와 학습을 진행합니다.")
            config = tomllib.load(fp)
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_synth_kwargs(self):
        """TTGANSynthesizer에 전달할 학습 관련 파라미터 dict"""
        return {
            'embedding_dim': self.embedding_dim,
            'generator_num_layers': self.generator_num_layers,
            'discriminator_dim': tuple(self.discriminator_dim),
            'generator_lr': self.generator_lr,
            'generator_decay': self.generator_decay,
            'discriminator_lr': self.discriminator_lr,
            'discriminator_decay': self.discriminator_decay,
            'batch_size': self.batch_size,
            'discriminator_steps': self.discriminator_steps,
            'log_frequency': self.log_frequency,
            'pac': self.pac,
            'cuda': self.cuda,
            'gradient_penalty_lambda': self.gradient_penalty_lambda,
            'cond_loss_weight': self.cond_loss_weight,
            'use_dynamic_weights': self.use_dynamic_weights,
            'dynamic_weight_start': self.dynamic_weight_start,
            'dynamic_weight_end': self.dynamic_weight_end,
            'use_kl_anneal': self.use_kl_anneal,
            'use_lr_scheduler': self.use_lr_scheduler,
            'use_r1_penalty': self.use_r1_penalty,
            'r1_weight': self.r1_weight,
            'use_generator_ema': self.use_generator_ema,
            'ema_decay': self.ema_decay,
            'use_mixed_precision': self.use_mixed_precision,
            'grad_clip_norm': self.grad_clip_norm,
            'use_label_smoothing': self.use_label_smoothing,
            'label_smoothing': self.label_smoothing,
            'use_residual_discriminator': self.use_residual_discriminator,
            'discriminator_residual_layers': self.discriminator_residual_layers,
            'discriminator_residual_dropout': self.discriminator_residual_dropout,
        }

    def sampling_options(self):
        """샘플링 단계에서 사용할 옵션 dict"""
        return {
            'enable_sampling_noise': self.enable_sampling_noise,
            'sampling_noise_std_ratio': self.sampling_noise_std_ratio,
            'use_discretized_rounding_logic': self.use_discretized_rounding_logic,
        }
