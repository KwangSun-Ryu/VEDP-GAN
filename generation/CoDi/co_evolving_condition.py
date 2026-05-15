import logging
import math
import os

from absl import flags
from tqdm.auto import tqdm
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .tabular_dataload import get_dataset
from .diffusion_continuous import GaussianDiffusionSampler, GaussianDiffusionTrainer
from .diffusion_discrete import MultinomialDiffusion
from .models.tabular_unet import tabularUnet
from .utils import (
    apply_activate,
    infiniteloop,
    log_sample_categorical,
    make_negative_condition,
    sampling_with,
    training_with,
    warmup_lr )

def _train_single_run(
    class_key,
    train_con_np,
    train_dis_np,
    exp_dir,
    device,
    num_class,
    transformer_dis,
    FLAGS,
    checkpoint_name ):
    
    os.makedirs(exp_dir, exist_ok=True)

    samples, con_dim = train_con_np.shape
    _, dis_dim = train_dis_np.shape

    steps_per_epoch = math.ceil(samples / FLAGS.training_batch_size)
    total_steps = FLAGS.total_epochs_both * steps_per_epoch

    logging.info(
        "[%s] samples: %d, con_dim: %d, dis_dim: %d, total_steps: %d",
        class_key,
        samples,
        con_dim,
        dis_dim,
        total_steps )

    train_iter_con = DataLoader(train_con_np, batch_size=FLAGS.training_batch_size, shuffle=True)
    train_iter_dis = DataLoader(train_dis_np, batch_size=FLAGS.training_batch_size, shuffle=True)
    datalooper_train_con = infiniteloop(train_iter_con)
    datalooper_train_dis = infiniteloop(train_iter_dis)

    # Continuous diffusion model
    FLAGS.input_size = con_dim
    FLAGS.cond_size = dis_dim
    FLAGS.output_size = con_dim
    FLAGS.encoder_dim = list(map(int, FLAGS.encoder_dim_con.split(',')))
    FLAGS.nf = FLAGS.nf_con
    model_con = tabularUnet(FLAGS).to(device)
    has_con = con_dim > 0
    if has_con:
        optim_con = torch.optim.Adam(model_con.parameters(), lr=FLAGS.lr_con)
        sched_con = torch.optim.lr_scheduler.LambdaLR(optim_con, lr_lambda=warmup_lr)
        trainer = GaussianDiffusionTrainer(model_con, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    else:
        optim_con = None
        sched_con = None
        trainer = None

    # Discrete diffusion model
    FLAGS.input_size = dis_dim
    FLAGS.cond_size = con_dim
    FLAGS.output_size = dis_dim
    FLAGS.encoder_dim = list(map(int, FLAGS.encoder_dim_dis.split(',')))
    FLAGS.nf = FLAGS.nf_dis
    model_dis = tabularUnet(FLAGS).to(device)
    optim_dis = torch.optim.Adam(model_dis.parameters(), lr=FLAGS.lr_dis)
    sched_dis = torch.optim.lr_scheduler.LambdaLR(optim_dis, lr_lambda=warmup_lr)
    trainer_dis = MultinomialDiffusion(
        num_class,
        train_dis_np.shape,
        model_dis,
        FLAGS,
        timesteps=FLAGS.T,
        loss_type='vb_stochastic',
    ).to(device)

    if FLAGS.parallel and has_con:
        trainer = torch.nn.DataParallel(trainer)

    num_params_con = sum(p.numel() for p in model_con.parameters())
    num_params_dis = sum(p.numel() for p in model_dis.parameters())
    logging.info(f'Continuous model params: {num_params_con:,}')
    logging.info(f'Discrete model params: {num_params_dis:,}')

    for step in tqdm(range(total_steps), desc='train', total=total_steps, leave=True):
        if has_con:
            model_con.train()
        model_dis.train()

        x_0_con = next(datalooper_train_con).to(device).float()
        x_0_dis = next(datalooper_train_dis).to(device).float()

        ns_con, ns_dis = make_negative_condition(x_0_con, x_0_dis)
        con_loss, con_loss_ns, dis_loss, dis_loss_ns = training_with(
            x_0_con,
            x_0_dis,
            trainer,
            trainer_dis,
            ns_con,
            ns_dis,
            transformer_dis,
            FLAGS,
            has_con=has_con )

        loss_con = con_loss + FLAGS.lambda_con * con_loss_ns
        loss_dis = dis_loss + FLAGS.lambda_dis * dis_loss_ns

        if has_con:
            optim_con.zero_grad()
            loss_con.backward()
            torch.nn.utils.clip_grad_norm_(model_con.parameters(), FLAGS.grad_clip)
            optim_con.step()
            sched_con.step()

        optim_dis.zero_grad()
        loss_dis.backward()
        torch.nn.utils.clip_grad_value_(trainer_dis.parameters(), FLAGS.grad_clip)
        torch.nn.utils.clip_grad_norm_(trainer_dis.parameters(), FLAGS.grad_clip)
        optim_dis.step()
        sched_dis.step()

    checkpoint = {
        'model_con': model_con.state_dict(),
        'model_dis': model_dis.state_dict() }
    
    torch.save(checkpoint, os.path.join(exp_dir, checkpoint_name))
    logging.info(f"[{class_key}] Saved checkpoint: {checkpoint_name}")


def _build_model_pair(con_dim, dis_dim, num_class, ref_shape, FLAGS, device):
    """학습 단계와 동일한 구조를 재구성한다."""

    FLAGS.input_size = con_dim
    FLAGS.cond_size = dis_dim
    FLAGS.output_size = con_dim
    FLAGS.encoder_dim = list(map(int, FLAGS.encoder_dim_con.split(',')))
    FLAGS.nf = FLAGS.nf_con
    model_con = tabularUnet(FLAGS).to(device)
    net_sampler = GaussianDiffusionSampler(
        model_con,
        FLAGS.beta_1,
        FLAGS.beta_T,
        FLAGS.T,
        FLAGS.mean_type,
        FLAGS.var_type,
    ).to(device)

    FLAGS.input_size = dis_dim
    FLAGS.cond_size = con_dim
    FLAGS.output_size = dis_dim
    FLAGS.encoder_dim = list(map(int, FLAGS.encoder_dim_dis.split(',')))
    FLAGS.nf = FLAGS.nf_dis
    model_dis = tabularUnet(FLAGS).to(device)
    trainer_dis = MultinomialDiffusion(
        num_class,
        ref_shape,
        model_dis,
        FLAGS,
        timesteps=FLAGS.T,
        loss_type='vb_stochastic',
    ).to(device)

    return model_con, model_dis, net_sampler, trainer_dis


def _mapping_key(value):
    """mappings.json 키와 일치하도록 값을 문자열로 변환한다."""

    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return str(int(round(float(value))))
    return str(value)


def _apply_mappings_to_df(df, mappings, class_key):
    """mappings.json 정보를 사용해 범주형 값을 복원한다."""

    target_info = mappings.get('target', {})
    target_col = target_info.get('column')
    target_map = target_info.get('mapping', {})

    for col, map_dict in mappings.get('categorical', {}).items():
        if col in df.columns:
            df[col] = df[col].map(lambda x: map_dict.get(_mapping_key(x), x))

    if target_col and target_col in df.columns and target_map:
        class_label = class_key.split('_')[-1]
        mapped_label = target_map.get(class_label, target_map.get(_mapping_key(class_label)))
        if mapped_label is not None:
            df[target_col] = mapped_label
        else:
            df[target_col] = df[target_col].map(lambda x: target_map.get(_mapping_key(x), x))

    return df


def _generate_class_samples(
    class_key,
    sample_size,
    checkpoint_path,
    con_dim,
    dis_dim,
    num_class,
    ref_shape,
    transformer_con,
    transformer_dis,
    con_idx,
    dis_idx,
    column_defs,
    FLAGS,
    device,
):
    """특정 클래스 모델로부터 합성 데이터를 생성한다."""

    column_names = [col['name'] for col in column_defs]
    column_count = len(column_defs)
    if sample_size <= 0:
        return pd.DataFrame(columns=column_names)

    model_con, model_dis, net_sampler, trainer_dis = _build_model_pair(
        con_dim,
        dis_dim,
        num_class,
        ref_shape,
        FLAGS,
        device,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_con.load_state_dict(checkpoint['model_con'])
    model_dis.load_state_dict(checkpoint['model_dis'])
    model_con.eval()
    model_dis.eval()
    net_sampler.eval()
    trainer_dis.eval()

    batch_results = []
    remaining = sample_size
    batch_size = min(FLAGS.training_batch_size, sample_size)

    while remaining > 0:
        current = min(batch_size, remaining)
        with torch.no_grad():
            x_T_con = torch.randn(current, con_dim, device=device)
            zero_logits = torch.zeros(current, dis_dim, device=device)
            log_x_T_dis = log_sample_categorical(zero_logits, num_class).to(device)

            x_con, x_dis = sampling_with(
                x_T_con,
                log_x_T_dis,
                net_sampler,
                trainer_dis,
                transformer_con,
                FLAGS,
            )

            x_dis = apply_activate(x_dis, transformer_dis.output_info)

        sample_con = transformer_con.inverse_transform(x_con.detach().cpu().numpy())
        sample_dis = transformer_dis.inverse_transform(x_dis.detach().cpu().numpy())

        merged = np.empty((current, column_count), dtype=object)
        for idx_pos, col_idx in enumerate(con_idx):
            merged[:, col_idx] = sample_con[:, idx_pos]
        for idx_pos, col_idx in enumerate(dis_idx):
            merged[:, col_idx] = sample_dis[:, idx_pos]

        batch_df = pd.DataFrame(merged, columns=column_names)
        batch_results.append(batch_df)
        remaining -= current

    return pd.concat(batch_results, ignore_index=True)

def _check_generated_data(df, mappings, class_sizes):
    """
    생성된 데이터의 target 열이 올바른 라벨 분포를 따르는지 검사하는 함수

    예) class_0 전용 모델 → 생성된 모든 샘플의 target 값이 0 이어야 함
        class_1 전용 모델 → 생성된 모든 샘플의 target 값이 1 이어야 함
    """
    label_to_class = {0: 'class_0', 1: 'class_1'}
    target_col = mappings['target']['column']  # target 컬럼명 가져오기

    vals = df[target_col].value_counts()

    for label, count in sorted(vals.items()):
        expected = class_sizes[label_to_class[int(label)]]
        if expected != count:
            raise ValueError(
                f"❌ target 값 불일치: {label_to_class[int(label)]} "
                f"(expected {expected}, got {count})" )


def train(FLAGS):
    FLAGS = flags.FLAGS
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp_dir = os.path.join(FLAGS.exp_dir, FLAGS.data)

    (   _train_raw,
        train_con_data,
        train_dis_data,
        _test,
        (_transformer_con, transformer_dis, _meta), _con_idx, _dis_idx,
    ) = get_dataset(FLAGS)

    num_class = np.array([info[0] for info in transformer_dis.output_info])

    if FLAGS.mode == 'eval':
        raise ValueError('평가 기능은 삭제되었습니다. 학습 모드만 지원합니다.')

    if FLAGS.is_balanced:
        for class_key in sorted(train_con_data.keys()):
            if class_key not in train_dis_data:
                raise KeyError(f'{class_key} 데이터가 범주형 배열에 없습니다.')

            suffix = class_key.split('_')[-1]
            checkpoint_name = f'checkpoint_class_{suffix}.pt'
            _train_single_run(
                class_key,
                train_con_data[class_key],
                train_dis_data[class_key],
                exp_dir,
                device,
                num_class,
                transformer_dis,
                FLAGS,
                checkpoint_name )
    
    else:
        _train_single_run(
            'all',
            train_con_data['all'],
            train_dis_data['all'],
            exp_dir,
            device,
            num_class,
            transformer_dis,
            FLAGS,
            'checkpoint.pt' )


def sample(FLAGS, save=True, output_path=None, verbose=True):

    if not FLAGS.is_balanced:
        raise ValueError('현재 샘플링은 is_balanced=True 환경에서만 지원됩니다.')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(FLAGS.data_dir, FLAGS.data)
    exp_dir = os.path.join(FLAGS.exp_dir, FLAGS.data)
    if save:
        os.makedirs(FLAGS.save_dir, exist_ok=True)

    config_path = os.path.join(data_dir, f"{FLAGS.data}.json")
    mapping_path = os.path.join(data_dir, 'mappings.json')

    with open(config_path, 'r', encoding="utf-8") as file:
        config = json.load(file)

    original_size = config.get('original_data_size')
    if original_size is None:
        raise ValueError(f"original_data_size 정보가 {FLAGS.data}.json에 없습니다.")

    columns_order = config.get('columns_order', [col['name'] for col in config['columns']])

    with open(mapping_path, 'r', encoding="utf-8") as file:
        mappings = json.load(file)

    (
        _train_dict,
        train_con_data,
        train_dis_data,
        _test,
        (transformer_con, transformer_dis, meta),
        con_idx,
        dis_idx 
    ) = get_dataset(FLAGS)

    num_class = np.array([info[0] for info in transformer_dis.output_info])

    ref_shapes = {
        key: train_dis_data[key].shape for key in train_dis_data
    }

    con_dim = train_con_data['class_0'].shape[1]
    dis_dim = train_dis_data['class_0'].shape[1]

    per_class = original_size // 2
    class_sizes = {
        'class_0': per_class,
        'class_1': original_size - per_class,
    }

    synthetic_parts = []
    for class_key in ['class_0', 'class_1']:
        sample_size = class_sizes.get(class_key, 0)
        if sample_size <= 0:
            continue

        suffix = class_key.split('_')[-1]
        checkpoint_path = os.path.join(exp_dir, f'checkpoint_class_{suffix}.pt')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'{checkpoint_path} 파일을 찾을 수 없습니다.')

        ref_shape = ref_shapes.get(class_key, next(iter(ref_shapes.values())))
        class_df = _generate_class_samples(
            class_key,
            sample_size,
            checkpoint_path,
            con_dim,
            dis_dim,
            num_class,
            ref_shape,
            transformer_con,
            transformer_dis,
            con_idx,
            dis_idx,
            meta['columns'],
            FLAGS,
            device,
        )

        class_df = _apply_mappings_to_df(class_df, mappings, class_key)
        class_df = class_df[columns_order]
        synthetic_parts.append(class_df)

    if not synthetic_parts:
        raise RuntimeError('생성된 합성 데이터가 없습니다.')

    synthetic_df = pd.concat(synthetic_parts, ignore_index=True)
    
    # 생성된 데이터가 target 열이 올바른 라벨 분포를 따르는지 검사
    _check_generated_data(synthetic_df, mappings, class_sizes)
    
    if save:
        if output_path is not None:
            save_path = output_path
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        else:
            save_path = os.path.join(FLAGS.save_dir, f"{FLAGS.data}_CoDi_syn.csv")
        synthetic_df.to_csv(save_path, index=False)
        if verbose:
            logging.info('합성 데이터를 저장했습니다: %s', save_path)

    return synthetic_df
