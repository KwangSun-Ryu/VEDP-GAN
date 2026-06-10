"""Checkpointable CTGAN wrapper."""

import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch import optim
from tqdm import tqdm

from ctgan import CTGAN
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import random_state
from ctgan.synthesizers.ctgan import Discriminator, Generator
from sdv.single_table.ctgan import CTGANSynthesizer, _validate_no_category_dtype
from sdv.single_table.utils import detect_discrete_columns

from generation.selection import should_save_candidate


class CheckpointableCTGAN(CTGAN):
    def __init__(self, *args, checkpoint_dir=None, candidate_start_epoch=None,
                 selection_save_every=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._selection_checkpoint_dir = checkpoint_dir
        self._selection_candidate_start_epoch = candidate_start_epoch
        self._selection_save_every = selection_save_every

    def _save_generator_candidate(self, epoch, total_epochs):
        if self._selection_checkpoint_dir is None:
            return
        if not should_save_candidate(
            epoch,
            self._selection_candidate_start_epoch,
            self._selection_save_every,
            total_epochs,
        ):
            return
        candidate_dir = os.path.join(self._selection_checkpoint_dir, f"epoch_{epoch:04d}")
        os.makedirs(candidate_dir, exist_ok=True)
        torch.save(self._generator.state_dict(), os.path.join(candidate_dir, "generator.pt"))

    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        self._validate_discrete_columns(train_data, discrete_columns)
        self._validate_null_data(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                "`epochs` argument in `fit` method has been deprecated.",
                DeprecationWarning,
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)
        train_data = self._transformer.transform(train_data)
        self._data_sampler = DataSampler(
            train_data, self._transformer.output_info_list, self._log_frequency
        )

        data_dim = self._transformer.output_dimensions
        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim,
        ).to(self._device)
        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac,
        ).to(self._device)

        optimizer_g = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )
        optimizer_d = optim.Adam(
            discriminator.parameters(), lr=self._discriminator_lr, betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1
        self.loss_values = pd.DataFrame(columns=["Epoch", "Generator Loss", "Distriminator Loss"])
        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = "Gen. ({gen:.2f}) | Discrim. ({dis:.2f})"
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for epoch_idx in epoch_iterator:
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
                    real = torch.from_numpy(real.astype("float32")).to(self._device)
                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        fake_cat = fakeact
                        real_cat = real

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)
                    pen = discriminator.calc_gradient_penalty(real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizer_d.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizer_d.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)
                if condvec is None:
                    c1, m1 = None, None
                else:
                    c1, m1, _, _ = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                y_fake = discriminator(torch.cat([fakeact, c1], dim=1)) if c1 is not None else discriminator(fakeact)
                cross_entropy = 0 if condvec is None else self._cond_loss(fake, c1, m1)
                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizer_g.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizer_g.step()

            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()
            epoch_loss_df = pd.DataFrame({
                "Epoch": [epoch_idx],
                "Generator Loss": [generator_loss],
                "Discriminator Loss": [discriminator_loss],
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df
            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )
            self._save_generator_candidate(epoch_idx + 1, epochs)

        if self._selection_checkpoint_dir is not None:
            last_dir = os.path.join(self._selection_checkpoint_dir, "last")
            os.makedirs(last_dir, exist_ok=True)
            torch.save(self._generator.state_dict(), os.path.join(last_dir, "generator.pt"))


class CheckpointableCTGANSynthesizer(CTGANSynthesizer):
    def __init__(self, *args, checkpoint_dir=None, candidate_start_epoch=None,
                 selection_save_every=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._checkpoint_dir = checkpoint_dir
        self._candidate_start_epoch = candidate_start_epoch
        self._selection_save_every = selection_save_every

    def _fit(self, processed_data):
        _validate_no_category_dtype(processed_data)
        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(self.metadata, processed_data, transformers)
        self._model = CheckpointableCTGAN(
            **self._model_kwargs,
            checkpoint_dir=self._checkpoint_dir,
            candidate_start_epoch=self._candidate_start_epoch,
            selection_save_every=self._selection_save_every,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS.*")
            self._model.fit(processed_data, discrete_columns=discrete_columns)
