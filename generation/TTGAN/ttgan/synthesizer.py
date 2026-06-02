"""TTGAN 모델을 감싸는 래퍼 모듈."""

import pandas as pd
import numpy as np

from generation.TTGAN.ttgan.ttgan import TTGAN

from sdv.single_table.base import BaseSingleTableSynthesizer
from sdv.single_table.utils import detect_discrete_columns

from tqdm.auto import tqdm

class TTGANSynthesizer(BaseSingleTableSynthesizer):
    """``TTGAN`` 모델을 감싸는 클래스.

    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            Single table metadata representing the data that this synthesizer will be used for.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        enforce_rounding (bool):
            Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
            by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers. Defaults to ``None``.
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original TTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool or str):
            If ``True``, use CUDA. If a ``str``, use the indicated device.
            If ``False``, do not use cuda at all.
    """

    _model_sdtype_transformers = {'categorical': None}

    def __init__(self, metadata, enforce_min_max_values=True, enforce_rounding=True, locales=None,
                 target=None, embedding_dim=128, generator_num_layers=6, discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=1000, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True, checkpoint=None,
                 use_residual_discriminator=False, discriminator_residual_layers=0,
                 discriminator_residual_dropout=0.3, gradient_penalty_lambda=10.0,
                 cond_loss_weight=1.0, use_dynamic_weights=False, dynamic_weight_start=0.5,
                 dynamic_weight_end=1.0, use_kl_anneal=False, use_lr_scheduler=False,
                 use_r1_penalty=False, r1_weight=10.0, use_generator_ema=False, ema_decay=0.999,
                 use_mixed_precision=False, grad_clip_norm=1.0, use_label_smoothing=False,
                 label_smoothing=0.05, selection_candidate_start_epoch=None,
                 selection_save_every=None):

        super().__init__(
            metadata=metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            locales=locales
        )

        self.target = target
        self.embedding_dim = embedding_dim
        self.generator_num_layers = generator_num_layers
        self.discriminator_dim = discriminator_dim
        self.generator_lr = generator_lr
        self.generator_decay = generator_decay
        self.discriminator_lr = discriminator_lr
        self.discriminator_decay = discriminator_decay
        self.batch_size = batch_size
        self.discriminator_steps = discriminator_steps
        self.log_frequency = log_frequency
        self.verbose = verbose
        self.epochs = epochs
        self.pac = pac
        self.cuda = cuda
        self.checkpoint = checkpoint
        self.use_residual_discriminator = use_residual_discriminator
        self.discriminator_residual_layers = discriminator_residual_layers
        self.discriminator_residual_dropout = discriminator_residual_dropout
        self.gradient_penalty_lambda = gradient_penalty_lambda
        self.cond_loss_weight = cond_loss_weight
        self.use_dynamic_weights = use_dynamic_weights
        self.dynamic_weight_start = dynamic_weight_start
        self.dynamic_weight_end = dynamic_weight_end
        self.use_kl_anneal = use_kl_anneal
        self.use_lr_scheduler = use_lr_scheduler
        self.use_r1_penalty = use_r1_penalty
        self.r1_weight = r1_weight
        self.use_generator_ema = use_generator_ema
        self.ema_decay = ema_decay
        self.use_mixed_precision = use_mixed_precision
        self.grad_clip_norm = grad_clip_norm
        self.use_label_smoothing = use_label_smoothing
        self.label_smoothing = label_smoothing
        self.selection_candidate_start_epoch = selection_candidate_start_epoch
        self.selection_save_every = selection_save_every

        self._model_kwargs = {
            'target': target,
            'embedding_dim': embedding_dim,
            'generator_num_layers': generator_num_layers,
            'discriminator_dim': discriminator_dim,
            'generator_lr': generator_lr,
            'generator_decay': generator_decay,
            'discriminator_lr': discriminator_lr,
            'discriminator_decay': discriminator_decay,
            'batch_size': batch_size,
            'discriminator_steps': discriminator_steps,
            'log_frequency': log_frequency,
            'verbose': verbose,
            'epochs': epochs,
            'pac': pac,
            'cuda': cuda,
            'checkpoint': checkpoint,
            'use_residual_discriminator': use_residual_discriminator,
            'discriminator_residual_layers': discriminator_residual_layers,
            'discriminator_residual_dropout': discriminator_residual_dropout,
            'gradient_penalty_lambda': gradient_penalty_lambda,
            'cond_loss_weight': cond_loss_weight,
            'use_dynamic_weights': use_dynamic_weights,
            'dynamic_weight_start': dynamic_weight_start,
            'dynamic_weight_end': dynamic_weight_end,
            'use_kl_anneal': use_kl_anneal,
            'use_lr_scheduler': use_lr_scheduler,
            'use_r1_penalty': use_r1_penalty,
            'r1_weight': r1_weight,
            'use_generator_ema': use_generator_ema,
            'ema_decay': ema_decay,
            'use_mixed_precision': use_mixed_precision,
            'grad_clip_norm': grad_clip_norm,
            'use_label_smoothing': use_label_smoothing,
            'label_smoothing': label_smoothing,
            'selection_candidate_start_epoch': selection_candidate_start_epoch,
            'selection_save_every': selection_save_every,
        }

    def _fit(self, processed_data):
        """모델을 데이터에 학습시킨다.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        transformers = self.get_transformers()
        discrete_columns = detect_discrete_columns(self.get_metadata(), processed_data, transformers)
        self._model = TTGAN(**self._model_kwargs)
        self._model.fit(processed_data, discrete_columns=discrete_columns)

    def _sample(self, num_rows, conditions=None):
        """모델에서 지정된 수만큼 샘플을 생성한다.

        Args:
            num_rows (int):
                Amount of rows to sample.
            conditions (dict):
                If specified, this dictionary maps column names to the column
                value. Then, this method generates ``num_rows`` samples, all of
                which are conditioned on the given variables.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        if conditions is None:
            return self._model.sample(num_rows)

        raise NotImplementedError("TTGANSynthesizer doesn't support conditional sampling.")

class TTGANWrapper:
    """``TTGANSynthesizer``를 여러 개 사용하여 클래스 별로 학습하는 Wrapper.

    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            Single table metadata representing the data that this synthesizer will be used for.
        target (str): Name of the target column.
        ckpt_path (str): checkpoint를 저장할 경로
        class_labels (list): target이 가지고 있는 class labels ex) [0, 1]
    """

    def __init__(self, metadata, target, class_labels, verbose, epochs, ckpt_path,
                 synth_kwargs=None, classwise=True):
        synth_kwargs = synth_kwargs or {}
        self.classwise = classwise
        self.target = target
        self.class_labels = class_labels
        self.single_model = None
        self.models = []

        if self.classwise:
            assert len(class_labels) == 2, "⚠️ class 개수는 반드시 2개이어야 합니다."
            self.models = [
                TTGANSynthesizer(metadata   = metadata,
                                 target     = str(label),
                                 verbose    = verbose,
                                 epochs     = epochs,
                                 checkpoint = ckpt_path,
                                 **synth_kwargs) for label in class_labels ]
        else:
            self.single_model = TTGANSynthesizer(
                metadata   = metadata,
                target     = target,
                verbose    = verbose,
                epochs     = epochs,
                checkpoint = ckpt_path,
                **synth_kwargs )
        
    
    def fit(self, data):
        if not self.classwise:
            self.single_model.fit(data)
            return
        with tqdm(total=len(self.class_labels), desc='label-wise fit',
                        position=0, leave=False, colour='#9b59b6') as pbar:
            for label, model in zip(self.class_labels, self.models):
                model.fit(data[data[self.target] == label])
                pbar.update(1)

    def sample(self, num_rows):
        """ num_rows를 받아 데이터를 생성함 """
        if not self.classwise:
            return self.single_model._sample(num_rows).copy()
        first_label, second_label = self.class_labels
        # 데이터 개수 지정
        num_first = num_rows // 2
        num_second = num_rows - num_first
        
        # 데이터 생성 순서
        iter_data = [
            (first_label, num_first, self.models[0]),
            (second_label, num_second, self.models[1]) ]
        
        samples = []
        for label, count, model in iter_data:
            df = model._sample(count).copy()
            samples.append(df)
        
        return pd.concat(samples).sample(frac=1, ignore_index=True) # 행 셔플
