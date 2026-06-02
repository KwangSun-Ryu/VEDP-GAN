from copy import deepcopy
import torch
import os
import numpy as np
from generation.TabDDPM import zero
from generation.TabDDPM import lib
from generation.TabDDPM.tab_ddpm import GaussianMultinomialDiffusion
from .utils_train import get_model, make_dataset, update_ema
import pandas as pd
from generation.selection import should_save_candidate

class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, device=torch.device('cuda:0'),
                 selection_enabled=False, candidate_start_step=None, selection_save_every=None,
                 checkpoints_dir=None):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps 
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.log_every = 100
        self.print_every = 1000
        self.ema_every = 1000
        self.selection_enabled = bool(selection_enabled)
        self.candidate_start_step = candidate_start_step
        self.selection_save_every = selection_save_every
        self.checkpoints_dir = checkpoints_dir

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def _save_candidate(self, step):
        if not self.selection_enabled or self.checkpoints_dir is None:
            return
        if not should_save_candidate(step, self.candidate_start_step, self.selection_save_every, self.steps):
            return
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        torch.save(
            self.diffusion._denoise_fn.state_dict(),
            os.path.join(self.checkpoints_dir, f"step_{step:05d}.pt"))
        torch.save(
            self.ema_model.state_dict(),
            os.path.join(self.checkpoints_dir, f"step_{step:05d}_ema.pt"))

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        while step < self.steps:
            x, out_dict = next(self.train_iter)
            out_dict = {'y': out_dict}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1):5d}/{self.steps:5d} MLoss: {mloss:.4f} GLoss: {gloss:.4f} Sum: {mloss + gloss:.4f}')
                self.loss_history.loc[len(self.loss_history)] =[step + 1, mloss, gloss, mloss + gloss]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1
            self._save_candidate(step)

def train(
    parent_dir,
    real_data_dir = 'data/TabDDPM_data/Gallstone',
    steps = 1000,
    lr = 0.002,
    weight_decay = 1e-4,
    batch_size = 1024,
    model_type = 'mlp',
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    device = torch.device('cuda:1'),
    seed = 0,
    change_val = False,
    selection_enabled = False,
    candidate_start_step = None,
    selection_save_every = None,
    checkpoints_dir = None ):
    
    real_data_dir = os.path.normpath(real_data_dir)
    parent_dir = os.path.normpath(parent_dir)

    zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)

    dataset = make_dataset(
        real_data_dir,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in
    
    model = get_model(model_type, model_params, num_numerical_features, category_sizes=dataset.get_category_sizes('train'))
    model.to(device)

    train_loader = lib.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)



    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device )
    
    diffusion.to(device)
    diffusion.train()

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device,
        selection_enabled=selection_enabled,
        candidate_start_step=candidate_start_step,
        selection_save_every=selection_save_every,
        checkpoints_dir=checkpoints_dir )

    trainer.run_loop()

    trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))
    if checkpoints_dir is not None:
        os.makedirs(checkpoints_dir, exist_ok=True)
        torch.save(diffusion._denoise_fn.state_dict(), os.path.join(checkpoints_dir, 'model_last.pt'))
        torch.save(trainer.ema_model.state_dict(), os.path.join(checkpoints_dir, 'model_last_ema.pt'))
