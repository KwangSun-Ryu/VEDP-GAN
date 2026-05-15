'''
dataloader를 불러와 데이터셋을 생성 후, 모델 학습 및 생성하는 클래스 생성하는 코드
'''
import json
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
import os
import shutil
import importlib.util
import subprocess
import sys
from types import SimpleNamespace

import torch
from wcwidth import wcswidth

from .dataloader import TabularDataLoader
from utils import set_seed

#### CTGAN 라이브러리 ####
from sdv.single_table import CTGANSynthesizer
from sdv.sampling import Condition
from sdv.utils import load_synthesizer

class Generator():
    def __init__(self, data_name, model_name, data_dir='./data', exp_dir='./exp', save_dir='./output', seed=42,
                 verbose=True, prepare_data=True, sampling_strategy='prior', config_path=None):
        self.verbose = bool(verbose)
        self.data_loader  = TabularDataLoader(
            data_name,
            model_name,
            data_dir,
            seed,
            prepare_data=prepare_data,
            verbose=self.verbose,
        ).make_dataloader()
        self.data_name    = data_name
        self.model_name   = model_name # 모델 이름
        self.data_dir     = data_dir   # 데이터셋 경로
        self.exp_dir      = exp_dir    # 모델 가중치 및 설정 파일 경로
        self.save_dir     = save_dir   # 데이터 저장 경로
        self.device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed         = set_seed(seed)
        self.sampling_strategy = sampling_strategy
        self.config_path = config_path or str(Path(__file__).resolve().parents[1] / 'config' / 'generation' / 'tadgan.toml')
        self.auto_trained = None
        self.tadgan_model = None
        self.target = json.loads(Path(os.path.join(self.data_dir, 'datasets_info.json')).read_text(encoding='utf-8'))[data_name]['target']

        os.makedirs(self.model_exp_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)
    
    @property
    def model_exp_dir(self):
        return os.path.join(self.exp_dir, self.model_name)
    
    @property
    def model_save_dir(self):
        return os.path.join(self.save_dir, self.model_name)
    
    def _terminal_width(self):
        return shutil.get_terminal_size(fallback=(80, 24)).columns

    def _banner(self, text):
        width = self._terminal_width()
        display_width = wcswidth(text)
        if display_width < 0:
            display_width = len(text)
        pad = max(0, width - display_width)
        left = pad // 2
        right = pad - left
        return "=" * left + text + "=" * right

    def _run_subprocess(self, command, verbose=True):
        run_kwargs = {'check': True}
        if not verbose:
            run_kwargs['stdout'] = subprocess.DEVNULL
            run_kwargs['stderr'] = subprocess.DEVNULL
        subprocess.run(command, **run_kwargs)

    @staticmethod
    @contextmanager
    def _suppress_output(enabled):
        if not enabled:
            yield
            return

        previous_tqdm_disable = os.environ.get('TQDM_DISABLE')
        os.environ['TQDM_DISABLE'] = '1'

        try:
            with open(os.devnull, 'w', encoding='utf-8') as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    yield
        finally:
            if previous_tqdm_disable is None:
                os.environ.pop('TQDM_DISABLE', None)
            else:
                os.environ['TQDM_DISABLE'] = previous_tqdm_disable

    @staticmethod
    def train_decorator(func):
        def wrapper(self, *args, **kwargs):
            verbose = kwargs.get('verbose', self.verbose)
            if verbose:
                print(self._banner(f" {self.model_name} 학습 시작 "))

            with self._suppress_output(enabled=not verbose):
                result = func(self, *args, **kwargs)

            if verbose:
                print(self._banner(f" {self.model_name} 학습 종료 "))
            return result
        return wrapper

    @staticmethod
    def inference_decorator(func):
        def wrapper(self, *args, **kwargs):
            verbose = kwargs.get('verbose', self.verbose)
            if verbose:
                print(self._banner(f" {self.model_name} 데이터 생성 시작 "))

            with self._suppress_output(enabled=not verbose):
                result = func(self, *args, **kwargs)

            if verbose:
                print(self._banner(f" {self.model_name} 데이터 생성 종료 "))
                print()
            return result
        return wrapper
    
    @train_decorator
    def train(self, verbose=True):
        ''' 모델 학습 '''
        if self.model_name == 'CTGAN':
            # 데이터 & 모델 불러오기
            generator = CTGANSynthesizer(metadata=self.data_loader.metadata,
                                         enforce_min_max_values=True,
                                         epochs=1000,
                                         cuda=True if self.device == 'cuda' else False,
                                         verbose=verbose)
            generator.fit(self.data_loader.train_data)
            # 모델 저장
            generator.save(os.path.join(self.model_exp_dir, f'{self.data_name}_{self.model_name}.pkl'))
            
            return generator
        
        if self.model_name == 'TabDDPM':
            module_path = 'generation.TabDDPM.scripts.pipeline'
            command = [
                sys.executable, '-m', module_path,
                '--data-name', self.data_name,
                '--data-dir', self.data_dir,
                '--exp-dir', self.model_exp_dir,
                '--save-dir', self.model_save_dir,
                '--train' ]
            
            self._run_subprocess(command, verbose=verbose)
            
            return None
        
        if self.model_name == 'STaSy':
            module_path = 'generation.STaSy.main'
            config_path = os.path.join(self.data_dir, f"{self.model_name}_data", self.data_name, f"{self.data_name}.py")
            command = [
                sys.executable, '-m', module_path,
                '--mode', 'train',
                '--config', config_path,
                '--data_dir', self.data_dir,
                '--exp_dir', self.model_exp_dir,
                '--save_dir', self.model_save_dir, 
                '--is_balanced=true' ]
            
            self._run_subprocess(command, verbose=verbose)
            
            return None
        
        if self.model_name == 'CoDi':
            module_path = 'generation.CoDi.main'
            codi_data_dir = os.path.join(self.data_dir, f"{self.model_name}_data")
            command = [
                sys.executable, '-m', module_path,
                '--mode', 'train',
                '--data_dir', codi_data_dir,
                '--exp_dir', self.model_exp_dir,
                '--save_dir', self.model_save_dir,
                '--data', self.data_name,
                '--is_balanced=true']
            
            self._run_subprocess(command, verbose=verbose)
            
            return None
        
        if self.model_name == 'AutoDiff':
            from generation.AutoDiff.scripts import pipeline as autodiff_pipeline

            auto_args = SimpleNamespace(
                data_name=self.data_name,
                model_name=self.model_name,
                exp_dir=self.model_exp_dir,
                save_dir=self.model_save_dir,
                device=self.device,
                dataloaders=self.data_loader )

            self.auto_trained = autodiff_pipeline._train_model(auto_args, self.data_loader, verbose=verbose)
            return self.auto_trained

        if self.model_name == 'TTGAN':
            module_path = 'generation.TTGAN.scripts.train_predictor'
            command = [
                sys.executable, '-m', module_path,
                '--data-dir',         self.data_dir,
                '--exp-dir',          self.exp_dir,
                '--data-name',        self.data_name,
                '--pred-model-name',  'LGBM',
                '--device',           'gpu',
                '--seed',             str(self.seed) ]
            
            self._run_subprocess(command, verbose=verbose)
            
            module_path = 'generation.TTGAN.scripts.train_generator'
            command = [
            sys.executable, '-m',   module_path, 
                '--data-dir',       self.data_dir,
                '--exp-dir',        self.exp_dir,
                '--data-name',      self.data_name,
                '--gen-model-name', 'TTGAN-CAT',
                '--seed',           str(self.seed) ]
            
            self._run_subprocess(command, verbose=verbose)
            
            return None

        if self.model_name == 'TADGAN':
            from ablation_study.scripts.train_tadgan import train as train_tadgan
            from ablation_study.scripts.utils import build_run_dirs_from_base, load_toml

            tadgan_args = SimpleNamespace(
                data_name  = self.data_name,
                model_name = self.model_name,
                variant_slug = 'TADGAN',
                exp_dir    = self.model_exp_dir,
                save_dir   = self.model_save_dir,
                device     = self.device,
                config_path = self.config_path,
                config_dict = load_toml(self.config_path),
                sampling_strategy = self.sampling_strategy,
                resume = False,
                verbose_model = verbose,
                dataloaders= self.data_loader )

            run_dirs = build_run_dirs_from_base(os.path.join(tadgan_args.exp_dir, tadgan_args.data_name))
            self.tadgan_model, _ = train_tadgan(tadgan_args, self.data_loader, run_dirs, verbose=verbose)
            return self.tadgan_model

    def _load_stasy_config(self):
        config_path = os.path.join(
            self.data_dir,
            'STaSy_data',
            self.data_name,
            f"{self.data_name}.py")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"STaSy config 파일이 없습니다: {config_path}")

        module_name = f"_stasy_config_{self.data_name}"
        spec = importlib.util.spec_from_file_location(module_name, config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"STaSy config import 실패: {config_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, 'get_config'):
            raise AttributeError(f"STaSy config에 get_config 함수가 없습니다: {config_path}")
        return module.get_config()

    def _build_codi_flags(self, seed):
        return SimpleNamespace(
            data=self.data_name,
            data_dir=os.path.join(self.data_dir, 'CoDi_data'),
            exp_dir=self.model_exp_dir,
            save_dir=self.model_save_dir,
            mode='sample',
            is_balanced=True,
            seed=seed,
            encoder_dim=None,
            encoder_dim_con='64,128,256',
            encoder_dim_dis='64,128,256',
            nf=None,
            nf_con=16,
            nf_dis=64,
            input_size=None,
            cond_size=None,
            output_size=None,
            activation='relu',
            training_batch_size=2100,
            eval_batch_size=2100,
            T=50,
            beta_1=0.00001,
            beta_T=0.02,
            lr_con=2e-03,
            lr_dis=2e-03,
            total_epochs_both=20000,
            grad_clip=1.0,
            parallel=False,
            sample_step=2000,
            mean_type='epsilon',
            var_type='fixedsmall',
            ns_method=0,
            lambda_con=0.2,
            lambda_dis=0.2,
        )

    @inference_decorator
    def inference(self, seed=42, save=True, output_path=None, verbose=True):
        # 합성 데이터 생성
        file_path = output_path or os.path.join(
            self.model_save_dir,
            f"{self.data_name}_{self.model_name}_syn.csv")

        if save:
            os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)

        if self.model_name == 'CTGAN':
            generator = load_synthesizer(os.path.join(self.model_exp_dir, f'{self.data_name}_{self.model_name}.pkl'))
            generator._set_random_state(seed)
            num_positive = self.data_loader.num_data // 2
            num_negative = self.data_loader.num_data - num_positive
            condition_positive = Condition(
                num_rows=num_positive,
                column_values={f'{self.data_loader.target}': 1})
            condition_negative = Condition(
                num_rows=num_negative,
                column_values={f'{self.data_loader.target}': 0})

            if save:
                synthetic_data = generator.sample_from_conditions(
                    conditions=[condition_positive, condition_negative],
                    output_file_path=file_path)
            else:
                synthetic_data = generator.sample_from_conditions(
                    conditions=[condition_positive, condition_negative])

            if save and verbose:
                print(f"✔️ {os.path.basename(file_path)} 생성 완료!")
            return synthetic_data

        if self.model_name == 'TabDDPM':
            from generation.TabDDPM.scripts import pipeline as tabddpm_pipeline

            synthetic_data = tabddpm_pipeline.run_sample(
                data_name=self.data_name,
                data_dir=self.data_dir,
                exp_dir=self.model_exp_dir,
                save_dir=self.model_save_dir,
                sample_seed=seed,
                change_val=False,
                save=save,
                output_path=file_path if save else None,
                verbose=verbose,
            )
            if save and verbose:
                print(f"✔️ {os.path.basename(file_path)} 생성 완료!")
            return synthetic_data

        if self.model_name == 'STaSy':
            from generation.STaSy import run_lib as stasy_run_lib

            config = self._load_stasy_config()
            synthetic_data = stasy_run_lib.sample(
                config,
                self.data_dir,
                self.model_exp_dir,
                self.model_save_dir,
                is_balanced=True,
                seed=seed,
                save=save,
                output_path=file_path if save else None,
                verbose=verbose,
            )
            if save and verbose:
                print(f"✔️ {os.path.basename(file_path)} 생성 완료!")
            return synthetic_data

        if self.model_name == 'CoDi':
            from generation.CoDi import co_evolving_condition as codi_sampling

            codi_flags = self._build_codi_flags(seed)
            synthetic_data = codi_sampling.sample(
                codi_flags,
                save=save,
                output_path=file_path if save else None,
                verbose=verbose,
            )
            if save and verbose:
                print(f"✔️ {os.path.basename(file_path)} 생성 완료!")
            return synthetic_data

        if self.model_name == 'AutoDiff':
            from generation.AutoDiff.scripts import pipeline as autodiff_pipeline

            auto_args = SimpleNamespace(
                data_name=self.data_name,
                model_name=self.model_name,
                exp_dir=self.model_exp_dir,
                save_dir=self.model_save_dir,
                device=self.device,
                dataloaders=self.data_loader,
                seed=seed)

            model_input = self.auto_trained
            if model_input is None:
                ckpt_path = os.path.join(self.model_exp_dir, f"{self.data_name}_{self.model_name}.pt")
                model_input = ckpt_path

            synthetic_data = autodiff_pipeline._sample(
                auto_args,
                model_input,
                save=save,
                output_path=file_path if save else None,
                verbose=verbose,
            )
            if save and verbose:
                print(f"✔️ {os.path.basename(file_path)} 생성 완료!")
            return synthetic_data

        if self.model_name == 'TTGAN':
            from generation.TTGAN.scripts import sample as ttgan_sample

            ttgan_args = SimpleNamespace(
                data_name=self.data_name,
                pred_model_name='LGBM',
                gen_model_name='TTGAN-CAT',
                device='gpu',
                data_dir=self.data_dir,
                exp_dir=self.exp_dir,
                save_dir=self.save_dir,
                seed=seed,
                num_data=self.data_loader.num_data,
            )

            synthetic_data = ttgan_sample.sample(
                ttgan_args,
                save=save,
                output_path=file_path if save else None,
                verbose=verbose,
            )
            if save and verbose:
                print(f"✔️ {os.path.basename(file_path)} 생성 완료!")
            return synthetic_data

        if self.model_name == 'TADGAN':
            from ablation_study.scripts.sample_tadgan import sample as sample_tadgan
            from ablation_study.scripts.utils import build_run_dirs_from_base, load_toml

            tadgan_args = SimpleNamespace(
                data_name=self.data_name,
                model_name=self.model_name,
                variant_slug='TADGAN',
                exp_dir=self.model_exp_dir,
                save_dir=self.model_save_dir,
                device=self.device,
                dataloaders=self.data_loader,
                config_path=self.config_path,
                config_dict=load_toml(self.config_path),
                verbose_model=verbose,
                sampling_strategy=self.sampling_strategy,
            )
            run_dirs = build_run_dirs_from_base(os.path.join(tadgan_args.exp_dir, tadgan_args.data_name))

            model_input = self.tadgan_model
            if model_input is None:
                ckpt_path = os.path.join(
                    self.model_exp_dir,
                    self.data_name,
                    'checkpoints',
                    f"{self.data_name}_{self.model_name}.pt",
                )
                model_input = ckpt_path

            if isinstance(model_input, str):
                synthetic_path, synthetic_data = sample_tadgan(
                    tadgan_args, self.data_loader, run_dirs,
                    ckpt_path=model_input, return_frame=True, verbose=verbose)
            else:
                synthetic_path, synthetic_data = sample_tadgan(
                    tadgan_args, self.data_loader, run_dirs,
                    model=model_input, return_frame=True, verbose=verbose)
            if save and os.path.abspath(synthetic_path) != os.path.abspath(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                shutil.copy2(synthetic_path, file_path)
            if save and verbose:
                print(f"✔️ {os.path.basename(file_path)} 생성 완료!")
            return synthetic_data

        raise ValueError(f"지원되지 않는 모델입니다: {self.model_name}")
