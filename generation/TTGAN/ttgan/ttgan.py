"""TTGAN 모듈.

Transformer 기반 테이블 데이터 생성 모델의 핵심 구현을 포함한다.
"""

import copy
import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import (
    BatchNorm1d,
    Dropout,
    LayerNorm,
    LeakyReLU,
    Linear,
    Module,
    ReLU,
    Sequential,
    functional,
)
from torch.nn.utils import clip_grad_norm_
from torch import amp
from tqdm import tqdm

try:
    import torch.cuda.amp as cuda_amp  # type: ignore
except ImportError:  # pragma: no cover
    cuda_amp = None

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state


def _make_grad_scaler(use_amp, device_type: str):
    """AMP 설정에 맞춰 GradScaler를 생성함"""
    if use_amp:
        try:
            return amp.GradScaler(device_type, enabled=True)
        except (TypeError, ValueError):
            if device_type == "cuda" and cuda_amp is not None:
                return cuda_amp.GradScaler(enabled=True)
            return amp.GradScaler(enabled=True)
    try:
        return amp.GradScaler(device_type, enabled=False)
    except (TypeError, ValueError):
        if device_type == "cuda" and cuda_amp is not None:
            return cuda_amp.GradScaler(enabled=False)
        return amp.GradScaler(enabled=False)


class _AutocastAdaptor:
    """PyTorch 버전에 따라 자동으로 autocast 컨텍스트를 제공함"""

    def __init__(self, device_type: str, enabled: bool):
        self.device_type = device_type
        self.enabled = enabled
        self._ctx = None

    def __enter__(self):
        try:
            self._ctx = amp.autocast(self.device_type, enabled=self.enabled)
        except (TypeError, ValueError):
            if self.device_type == "cuda" and cuda_amp is not None:
                self._ctx = cuda_amp.autocast(enabled=self.enabled)
            else:
                self._ctx = torch.autocast(self.device_type, enabled=self.enabled)
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._ctx.__exit__(exc_type, exc_val, exc_tb)


class ResidualDiscriminatorBlock(Module):
    """Discriminator용 residual 블록"""

    def __init__(self, in_dim, out_dim, dropout):
        super(ResidualDiscriminatorBlock, self).__init__()
        self.fc1 = Linear(in_dim, out_dim)
        self.norm1 = LayerNorm(out_dim)
        self.act = LeakyReLU(0.2)
        self.dropout = Dropout(dropout)
        self.fc2 = Linear(out_dim, out_dim)
        self.norm2 = LayerNorm(out_dim)
        self.match = Linear(in_dim, out_dim) if in_dim != out_dim else None

    def forward(self, input_):
        """Residual block forward"""
        residual = input_ if self.match is None else self.match(input_)
        out = self.fc1(input_)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.norm2(out)
        out = out + residual
        out = self.act(out)
        return out


class Discriminator(Module):
    """TTGAN에서 사용되는 판별자 네트워크."""

    def __init__(self, input_dim, discriminator_dim, pac=10, use_residual=False,
                 residual_layers=0, residual_dropout=0.3):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        discriminator_dim = list(discriminator_dim)
        for item in discriminator_dim:
            if use_residual:
                seq.append(ResidualDiscriminatorBlock(dim, item, residual_dropout))
                for _ in range(max(0, residual_layers)):
                    seq.append(ResidualDiscriminatorBlock(item, item, residual_dropout))
            else:
                seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):
    """Residual layer for the TTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """TTGAN에서 실제 데이터를 생성하는 생성자 네트워크."""

    def __init__(self, embedding_dim, num_layers, data_dim):
        super(Generator, self).__init__()
        seq = [
            torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(
                    d_model = embedding_dim,
                    nhead   = self.optimal_nhead(embedding_dim),
                ),
                num_layers  = num_layers,
            )
        ]
        seq.append(Linear(embedding_dim, data_dim))
        self.seq = Sequential(*seq)

    def optimal_nhead(self, dim):
        """Calculate the optimal number of heads for the Transformer."""
        nhead = int(np.sqrt(dim))
        while True:
            if dim % nhead == 0:
                return nhead
            nhead -= 1

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data


class TTGAN(BaseSynthesizer):
    """ conditional Tabular GAN 생성기.

    TTGAN 프로젝트의 핵심 클래스이며 여러 구성 요소를 통합하여 동작한다.

    Args:
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
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(self, target=None, embedding_dim=128, generator_num_layers=6, discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True, checkpoint=None,
                 use_residual_discriminator=False, discriminator_residual_layers=0,
                 discriminator_residual_dropout=0.3, gradient_penalty_lambda=10.0,
                 cond_loss_weight=1.0, use_dynamic_weights=False, dynamic_weight_start=0.5,
                 dynamic_weight_end=1.0, use_kl_anneal=False, use_lr_scheduler=False,
                 use_r1_penalty=False, r1_weight=10.0, use_generator_ema=False, ema_decay=0.999,
                 use_mixed_precision=False, grad_clip_norm=1.0, use_label_smoothing=False,
                 label_smoothing=0.05):

        assert batch_size % 2 == 0

        self._target = target

        self._embedding_dim = embedding_dim
        self._generator_num_layers = generator_num_layers
        self._discriminator_dim = discriminator_dim
        self._use_residual_discriminator = use_residual_discriminator
        self._discriminator_residual_layers = discriminator_residual_layers
        self._discriminator_residual_dropout = discriminator_residual_dropout
        self._gp_lambda = gradient_penalty_lambda
        self._cond_loss_weight = cond_loss_weight
        self._use_dynamic_weights = use_dynamic_weights
        self._dynamic_weight_start = dynamic_weight_start
        self._dynamic_weight_end = dynamic_weight_end
        self._use_kl_anneal = use_kl_anneal
        self._use_lr_scheduler = use_lr_scheduler
        self._use_r1_penalty = use_r1_penalty
        self._r1_weight = r1_weight
        self._use_generator_ema = use_generator_ema
        self._ema_decay = ema_decay
        self._use_mixed_precision = use_mixed_precision
        self._grad_clip_norm = grad_clip_norm
        self._use_label_smoothing = use_label_smoothing
        self._label_smoothing = label_smoothing

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac
        self._checkpoint = checkpoint

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self._ema_generator_state = None

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """오래된 torch 버전에서 gumbel_softmax 불안정 문제를 처리하기 위한 함수.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m, label_smoothing=0.0):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none',
                        label_smoothing=label_smoothing,
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')
        
    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Fit the TTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency
        )

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_num_layers,
            data_dim
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac,
            use_residual=self._use_residual_discriminator,
            residual_layers=self._discriminator_residual_layers,
            residual_dropout=self._discriminator_residual_dropout,
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        optimizerD = optim.Adam(
            discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        scheduler_g = scheduler_d = None
        if self._use_lr_scheduler:
            scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizerG, T_max=epochs, eta_min=self._generator_lr * 0.1)
            scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizerD, T_max=epochs, eta_min=self._discriminator_lr * 0.1)

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)

        use_amp = self._use_mixed_precision and self._device.type == 'cuda'
        amp_device_type = 'cuda' if self._device.type == 'cuda' else 'cpu'
        scaler_d = _make_grad_scaler(use_amp, amp_device_type)
        scaler_g = _make_grad_scaler(use_amp, amp_device_type)

        ema_generator = None
        if self._use_generator_ema:
            ema_generator = copy.deepcopy(self._generator)
            ema_generator.to(self._device)
            ema_generator.eval()

            def _update_ema(target: torch.nn.Module, source: torch.nn.Module, decay: float):
                with torch.no_grad():
                    for ema_param, src_param in zip(target.parameters(), source.parameters()):
                        ema_param.data.mul_(decay).add_(src_param.data, alpha=1.0 - decay)
        else:
            def _update_ema(*args, **kwargs):
                return None

        for epoch_idx in epoch_iterator:
            running = {"g": 0.0, "d": 0.0}
            epoch_ratio = (epoch_idx + 1) / max(1, epochs)
            dynamic_factor = self._dynamic_weight_start + (self._dynamic_weight_end - self._dynamic_weight_start) * epoch_ratio if self._use_dynamic_weights else 1.0
            current_gp_lambda = self._gp_lambda * dynamic_factor
            cond_weight = self._cond_loss_weight * dynamic_factor if self._use_dynamic_weights else self._cond_loss_weight
            if self._use_kl_anneal:
                warmup = max(1, int(epochs * 0.3))
                cond_weight = cond_weight * min(1.0, (epoch_idx + 1) / warmup)

            for _ in range(steps_per_epoch):

                for _ in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(train_data, self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    with _AutocastAdaptor(amp_device_type, use_amp):
                        real_input = real_cat.requires_grad_(self._use_r1_penalty)
                        y_fake = discriminator(fake_cat)
                        y_real = discriminator(real_input)
                        loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                        pen = discriminator.calc_gradient_penalty(
                            real_cat, fake_cat, self._device, self.pac, lambda_=current_gp_lambda)
                        loss_d = loss_d + pen
                        if self._use_r1_penalty:
                            grad_real = torch.autograd.grad(
                                outputs=y_real.sum(),
                                inputs=real_input,
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True,
                            )[0]
                            grad_real = grad_real.view(grad_real.size(0), -1).float()
                            r1_penalty = 0.5 * self._r1_weight * (grad_real.norm(2, dim=1) ** 2).mean()
                            loss_d = loss_d + r1_penalty

                    optimizerD.zero_grad(set_to_none=False)
                    if use_amp:
                        scaler_d.scale(loss_d).backward()
                        scaler_d.unscale_(optimizerD)
                        if self._grad_clip_norm > 0:
                            clip_grad_norm_(discriminator.parameters(), self._grad_clip_norm)
                        scaler_d.step(optimizerD)
                        scaler_d.update()
                    else:
                        loss_d.backward()
                        if self._grad_clip_norm > 0:
                            clip_grad_norm_(discriminator.parameters(), self._grad_clip_norm)
                        optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    disc_input = torch.cat([fakeact, c1], dim=1)
                else:
                    disc_input = fakeact

                with _AutocastAdaptor(amp_device_type, use_amp):
                    y_fake = discriminator(disc_input)
                    if condvec is None:
                        cond_loss = torch.tensor(0.0, device=self._device)
                    else:
                        smoothing = self._label_smoothing if self._use_label_smoothing else 0.0
                        cond_loss = self._cond_loss(fake, c1, m1, label_smoothing=smoothing)
                    loss_g = -torch.mean(y_fake)
                    if cond_loss is not None:
                        loss_g = loss_g + cond_weight * cond_loss

                optimizerG.zero_grad(set_to_none=False)
                if use_amp:
                    scaler_g.scale(loss_g).backward()
                    scaler_g.unscale_(optimizerG)
                    if self._grad_clip_norm > 0:
                        clip_grad_norm_(self._generator.parameters(), self._grad_clip_norm)
                    scaler_g.step(optimizerG)
                    scaler_g.update()
                else:
                    loss_g.backward()
                    if self._grad_clip_norm > 0:
                        clip_grad_norm_(self._generator.parameters(), self._grad_clip_norm)
                    optimizerG.step()

                if self._use_generator_ema and ema_generator is not None:
                    _update_ema(ema_generator, self._generator, self._ema_decay)

            generator_loss = loss_g.detach().cpu()
            discriminator_loss = loss_d.detach().cpu()
            running["g"] += generator_loss.item()
            running["d"] += discriminator_loss.item()

            epoch_loss_df = pd.DataFrame({
                'Epoch': [epoch_idx],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss]
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat(
                    [self.loss_values, epoch_loss_df]
                ).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(
                        gen=running["g"] / max(1, steps_per_epoch),
                        dis=running["d"] / max(1, steps_per_epoch),
                    )
                )

            if scheduler_g is not None:
                scheduler_g.step()
            if scheduler_d is not None:
                scheduler_d.step()

        if ema_generator is not None:
            self._generator.load_state_dict(ema_generator.state_dict())
            self._ema_generator_state = ema_generator.state_dict()

        if self._checkpoint:
            os.makedirs(self._checkpoint, exist_ok=True)
            torch.save(self._generator.state_dict(),
                       os.path.join(self._checkpoint, "generator.pt"))

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size)
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
