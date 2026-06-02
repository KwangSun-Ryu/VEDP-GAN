'''
dataloader를 불러와 데이터셋을 생성 후, 모델 학습 및 생성하는 클래스 생성하는 코드
'''
import json
import glob
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
import os
import shutil
import importlib.util
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import torch
from wcwidth import wcswidth

from .dataloader import TabularDataLoader
from utils import set_seed
from generation.selection import copytree_replace, flatten_config, load_model_selection_config, run_best_selection

#### CTGAN 라이브러리 ####
from sdv.single_table import CTGANSynthesizer
from sdv.sampling import Condition
from sdv.utils import load_synthesizer

class Generator():
    def __init__(self, data_name, model_name, data_dir='./data', exp_dir='./exp', save_dir='./output', seed=42,
                 verbose=True, prepare_data=True, sampling_strategy='prior',
                 config_path=None, eval_model_config_dir='./config/prediction'):
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
        self.config_path = config_path
        self.eval_model_config_dir = eval_model_config_dir
        self.auto_trained = None
        self.tadgan_model = None
        self.tadgan_ckpt_path = None
        self.tadgan_run_dirs = None
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

    def _build_tadgan_context(self):
        from generation.TADGAN.scripts import utils as tadgan_utils

        if self.model_name != "TADGAN":
            raise ValueError("TADGAN context is only available for model_name='TADGAN'.")
        if not torch.cuda.is_available():
            raise RuntimeError("Patched TADGAN must run with CUDA in the WSL TADGAN environment.")

        project_root = Path(__file__).resolve().parents[1]
        config_path = Path(self.config_path) if self.config_path else project_root / "config" / "generation" / "tadgan.toml"
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        if not config_path.exists():
            raise FileNotFoundError(f"TADGAN generation config가 없습니다: {config_path}")
        with open(config_path, "rb") as file:
            config_dict = tomllib.load(file)
        config_flat = tadgan_utils.flatten_config_dict(config_dict)
        if config_flat.get("enable_best_on_test_selection") is not True:
            raise ValueError("TADGAN 일반 generation은 checkpoint_selection.enable_best_on_test_selection=true가 필수입니다.")

        base_dir = os.path.join(self.model_exp_dir, self.data_name)
        run_dirs = {
            "base_dir": base_dir,
            "checkpoints_dir": os.path.join(base_dir, "checkpoints"),
            "synthetic_dir": self.model_save_dir,
            "metrics_dir": os.path.join(base_dir, "metrics"),
            "logs_dir": os.path.join(base_dir, "logs"),
        }
        for path in run_dirs.values():
            os.makedirs(path, exist_ok=True)

        args = SimpleNamespace(
            experiment="generation",
            data_name=self.data_name,
            data_dir=self.data_dir,
            model_name="TADGAN",
            variant_slug="TADGAN",
            display_name="TADGAN",
            config_path=str(config_path),
            config_dict=config_dict,
            device_train="cuda",
            device=torch.device("cuda"),
            eval_model_config_dir=self.eval_model_config_dir,
            eval_model_num_trials=100,
            eval_stages=["ml", "sdmetrics"],
            test=False,
            test_num=10,
            device_ml="gpu",
            device_dcr="gpu",
            eval_ml_seed=self.seed,
            ml_eval_seed_base=self.seed,
            seed=self.seed,
            verbose_model=self.verbose,
            verbose_eval=False,
            multiprocessing=False,
            resume=False,
            num_workers=config_flat.get("num_workers", 0),
            exp_dir=self.exp_dir,
            stability_num_seeds=1,
            sampling_strategy=self.sampling_strategy or config_flat.get("sampling_strategy", "prior"),
            ks_complement_method=str(config_flat.get("ks_complement_method", "asymp")).strip().lower(),
            enable_best_on_test_selection=True,
            selection_save_every=max(1, int(config_flat.get("selection_save_every", 1))),
            checkpoint_selection_policy=str(config_flat.get("checkpoint_selection_policy", "stable_fidelity_score")),
            checkpoint_selection_use_fidelity_gate=bool(config_flat.get("checkpoint_selection_use_fidelity_gate", True)),
            checkpoint_selection_ksc_gate_delta=config_flat.get("checkpoint_selection_ksc_gate_delta", 0.03),
            checkpoint_selection_tvc_gate_delta=config_flat.get("checkpoint_selection_tvc_gate_delta", 0.02),
            checkpoint_selection_auc_weight=config_flat.get("checkpoint_selection_auc_weight", 0.60),
            checkpoint_selection_ksc_weight=config_flat.get("checkpoint_selection_ksc_weight", 0.25),
            checkpoint_selection_tvc_weight=config_flat.get("checkpoint_selection_tvc_weight", 0.15),
            checkpoint_selection_use_stable_score=bool(config_flat.get("checkpoint_selection_use_stable_score", True)),
            checkpoint_selection_current_weight=config_flat.get("checkpoint_selection_current_weight", 0.50),
            checkpoint_selection_prev_weight=config_flat.get("checkpoint_selection_prev_weight", 0.25),
            checkpoint_selection_next_weight=config_flat.get("checkpoint_selection_next_weight", 0.25),
            selection_candidate_start_epoch=config_flat.get("selection_candidate_start_epoch", None),
        )
        return args, run_dirs

    def _selection_context(self):
        return SimpleNamespace(
            data_name=self.data_name,
            data_dir=self.data_dir,
            model_name=self.model_name,
            exp_dir=self.exp_dir,
            save_dir=self.save_dir,
            log_dir=os.path.join(self.exp_dir, self.model_name, self.data_name, "selection_logs"),
            eval_model_config_dir=self.eval_model_config_dir,
            seed=self.seed,
        )

    def _require_path(self, path, label):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label}이 없습니다: {path}")
        return path

    def _set_config_value(self, config, key, value):
        try:
            setattr(config, key, value)
        except AttributeError:
            config.unlock()
            setattr(config, key, value)
            config.lock()

    def _epoch_from_path(self, path):
        base = os.path.basename(os.path.normpath(path))
        digits = ''.join(ch for ch in base if ch.isdigit())
        return int(digits) if digits else None

    def _candidate_dirs(self, root):
        candidates = []
        for path in sorted(glob.glob(os.path.join(root, "epoch_*"))):
            epoch = self._epoch_from_path(path)
            if epoch is not None:
                candidates.append({"epoch": epoch, "path": path})
        return candidates

    def _candidate_files(self, root, pattern):
        candidates = []
        for path in sorted(glob.glob(os.path.join(root, pattern))):
            epoch = self._epoch_from_path(path)
            if epoch is not None:
                candidates.append({"epoch": epoch, "path": path})
        return candidates

    def _run_tabddpm_selection(self):
        from generation.TabDDPM.scripts import pipeline as tabddpm_pipeline

        config = load_model_selection_config("TabDDPM")
        checkpoints_dir = os.path.join(self.model_exp_dir, self.data_name, "checkpoints")
        candidates = self._candidate_files(checkpoints_dir, "step_*.pt")
        candidates = [item for item in candidates if not str(item["path"]).endswith("_ema.pt")]

        def _sample(candidate, output_path):
            return tabddpm_pipeline.run_sample(
                data_name=self.data_name,
                data_dir=self.data_dir,
                exp_dir=self.model_exp_dir,
                save_dir=self.model_save_dir,
                sample_seed=self.seed,
                save=True,
                output_path=output_path,
                model_path=candidate["path"],
                verbose=False,
            )

        def _promote(selected):
            best_path = os.path.join(checkpoints_dir, "model_best_on_test.pt")
            shutil.copy2(selected["ckpt_path"], best_path)
            selected_ema = selected["ckpt_path"].replace(".pt", "_ema.pt")
            if os.path.exists(selected_ema):
                shutil.copy2(selected_ema, os.path.join(checkpoints_dir, "model_best_on_test_ema.pt"))
            return best_path

        def _cleanup(best_path):
            keep = {
                os.path.abspath(best_path),
                os.path.abspath(best_path.replace(".pt", "_ema.pt")),
                os.path.abspath(os.path.join(checkpoints_dir, "model_last.pt")),
                os.path.abspath(os.path.join(checkpoints_dir, "model_last_ema.pt")),
            }
            for path in glob.glob(os.path.join(checkpoints_dir, "step_*.pt")):
                if os.path.abspath(path) not in keep:
                    os.remove(path)

        self.tabddpm_best_path = run_best_selection(
            self._selection_context(), config, candidates, _sample, _promote, _cleanup)

    def _run_ctgan_selection(self):
        config = load_model_selection_config("CTGAN")
        checkpoints_dir = os.path.join(self.model_exp_dir, self.data_name, "checkpoints")
        candidates = self._candidate_dirs(checkpoints_dir)

        def _sample(candidate, output_path):
            generator = load_synthesizer(os.path.join(self.model_exp_dir, f'{self.data_name}_{self.model_name}.pkl'))
            state_path = os.path.join(candidate["path"], "generator.pt")
            state = torch.load(state_path, map_location=generator._model._device, weights_only=True)
            generator._model._generator.load_state_dict(state)
            return self._sample_ctgan_synthesizer(generator, self.seed, output_path)

        def _promote(selected):
            return copytree_replace(selected["ckpt_path"], os.path.join(checkpoints_dir, "best_on_test"))

        def _cleanup(best_path):
            keep = {os.path.abspath(best_path), os.path.abspath(os.path.join(checkpoints_dir, "last"))}
            for path in glob.glob(os.path.join(checkpoints_dir, "epoch_*")):
                if os.path.abspath(path) not in keep:
                    shutil.rmtree(path, ignore_errors=True)

        run_best_selection(self._selection_context(), config, candidates, _sample, _promote, _cleanup)

    def _sample_ctgan_synthesizer(self, generator, seed, output_path):
        generator._set_random_state(seed)
        num_positive = self.data_loader.num_data // 2
        num_negative = self.data_loader.num_data - num_positive
        conditions = [
            Condition(num_rows=num_positive, column_values={f'{self.data_loader.target}': 1}),
            Condition(num_rows=num_negative, column_values={f'{self.data_loader.target}': 0}),
        ]
        return generator.sample_from_conditions(conditions=conditions, output_file_path=output_path)

    def _run_stasy_selection(self):
        from generation.STaSy import run_lib as stasy_run_lib

        config = load_model_selection_config("STaSy")
        checkpoints_dir = os.path.join(self.model_exp_dir, self.data_name, "checkpoints")
        candidates = self._candidate_dirs(os.path.join(checkpoints_dir, "candidates"))

        def _sample(candidate, output_path):
            stasy_config = self._load_stasy_config()
            try:
                stasy_config.checkpoint_override_dir = candidate["path"]
            except AttributeError:
                stasy_config.unlock()
                stasy_config.checkpoint_override_dir = candidate["path"]
                stasy_config.lock()
            return stasy_run_lib.sample(
                stasy_config,
                self.data_dir,
                self.model_exp_dir,
                self.model_save_dir,
                is_balanced=True,
                seed=self.seed,
                save=True,
                output_path=output_path,
                verbose=False,
            )

        def _promote(selected):
            return copytree_replace(selected["ckpt_path"], os.path.join(checkpoints_dir, "best_on_test"))

        def _cleanup(best_path):
            shutil.rmtree(os.path.join(checkpoints_dir, "candidates"), ignore_errors=True)

        run_best_selection(self._selection_context(), config, candidates, _sample, _promote, _cleanup)

    def _run_codi_selection(self):
        from generation.CoDi import co_evolving_condition as codi_sampling

        config = load_model_selection_config("CoDi")
        exp_data_dir = os.path.join(self.model_exp_dir, self.data_name)
        candidates = self._candidate_dirs(os.path.join(exp_data_dir, "candidates"))

        def _sample(candidate, output_path):
            codi_args = self._build_codi_flags(self.seed)
            codi_args.checkpoint_override_dir = candidate["path"]
            return codi_sampling.sample(codi_args, save=True, output_path=output_path, verbose=False)

        def _promote(selected):
            return copytree_replace(selected["ckpt_path"], os.path.join(exp_data_dir, "best_on_test"))

        def _cleanup(best_path):
            shutil.rmtree(os.path.join(exp_data_dir, "candidates"), ignore_errors=True)

        run_best_selection(self._selection_context(), config, candidates, _sample, _promote, _cleanup)

    def _run_autodiff_selection(self):
        from generation.AutoDiff.scripts import pipeline as autodiff_pipeline

        config = load_model_selection_config("AutoDiff")
        checkpoints_dir = os.path.join(self.model_exp_dir, self.data_name, "checkpoints")
        candidates = self._candidate_files(checkpoints_dir, "epoch_*.pt")

        def _sample(candidate, output_path):
            auto_args = SimpleNamespace(
                data_name=self.data_name,
                model_name=self.model_name,
                exp_dir=self.model_exp_dir,
                save_dir=self.model_save_dir,
                device=self.device,
                dataloaders=self.data_loader,
                seed=self.seed)
            return autodiff_pipeline._sample(auto_args, candidate["path"], save=True, output_path=output_path, verbose=False)

        def _promote(selected):
            best_path = os.path.join(checkpoints_dir, "best_on_test.pt")
            shutil.copy2(selected["ckpt_path"], best_path)
            return best_path

        def _cleanup(best_path):
            keep = {
                os.path.abspath(best_path),
                os.path.abspath(os.path.join(checkpoints_dir, "last.pt")),
            }
            for path in glob.glob(os.path.join(checkpoints_dir, "epoch_*.pt")):
                if os.path.abspath(path) not in keep:
                    os.remove(path)

        self.autodiff_best_path = run_best_selection(
            self._selection_context(), config, candidates, _sample, _promote, _cleanup)
        self.auto_trained = None

    def _run_ttgan_selection(self):
        from generation.TTGAN.scripts import sample as ttgan_sample

        config = load_model_selection_config("TTGAN")
        generator_dir = os.path.join(self.exp_dir, "TTGAN", self.data_name, "generators")
        candidates = self._candidate_dirs(os.path.join(generator_dir, "candidates"))

        def _sample(candidate, output_path):
            ttgan_args = SimpleNamespace(
                data_name=self.data_name,
                pred_model_name='LGBM',
                gen_model_name='TTGAN-CAT',
                device='gpu',
                data_dir=self.data_dir,
                exp_dir=self.exp_dir,
                save_dir=self.save_dir,
                seed=self.seed,
                num_data=self.data_loader.num_data,
                checkpoint_override_dir=candidate["path"],
            )
            return ttgan_sample.sample(ttgan_args, save=True, output_path=output_path, verbose=False)

        def _promote(selected):
            return copytree_replace(selected["ckpt_path"], os.path.join(generator_dir, "best_on_test"))

        def _cleanup(best_path):
            shutil.rmtree(os.path.join(generator_dir, "candidates"), ignore_errors=True)

        run_best_selection(self._selection_context(), config, candidates, _sample, _promote, _cleanup)

    def _run_best_checkpoint_selection(self):
        runners = {
            "TabDDPM": self._run_tabddpm_selection,
            "CTGAN": self._run_ctgan_selection,
            "STaSy": self._run_stasy_selection,
            "CoDi": self._run_codi_selection,
            "AutoDiff": self._run_autodiff_selection,
            "TTGAN": self._run_ttgan_selection,
        }
        runner = runners.get(self.model_name)
        if runner is not None:
            runner()

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
            from generation.ctgan_checkpoint import CheckpointableCTGANSynthesizer

            selection_flat = flatten_config(load_model_selection_config("CTGAN"))
            checkpoints_dir = os.path.join(self.model_exp_dir, self.data_name, "checkpoints")
            ctgan_epochs = selection_flat.get("epochs", 1000)
            # 데이터 & 모델 불러오기
            generator = CheckpointableCTGANSynthesizer(
                metadata=self.data_loader.metadata,
                enforce_min_max_values=True,
                epochs=ctgan_epochs,
                cuda=self.device.type == 'cuda',
                verbose=verbose,
                checkpoint_dir=checkpoints_dir,
                candidate_start_epoch=min(ctgan_epochs, selection_flat.get("selection_candidate_start_epoch", 501)),
                selection_save_every=selection_flat.get("selection_save_every", 50))
            generator.fit(self.data_loader.train_data)
            # 모델 저장
            generator.save(os.path.join(self.model_exp_dir, f'{self.data_name}_{self.model_name}.pkl'))
            self._run_best_checkpoint_selection()
            
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
            self._run_best_checkpoint_selection()
            
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
            self._run_best_checkpoint_selection()
            
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
            self._run_best_checkpoint_selection()
            
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
            self._run_best_checkpoint_selection()
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
            self._run_best_checkpoint_selection()
            
            return None

        if self.model_name == 'TADGAN':
            from generation.TADGAN.scripts import train as tadgan_train
            from generation.TADGAN.scripts.selection import run_best_selection

            tadgan_args, run_dirs = self._build_tadgan_context()
            self.tadgan_run_dirs = run_dirs

            self.tadgan_model, self.tadgan_ckpt_path = tadgan_train.train(
                tadgan_args,
                self.data_loader,
                run_dirs,
                reporter=None,
                verbose=verbose,
            )
            self.tadgan_ckpt_path = run_best_selection(tadgan_args, self.data_loader, run_dirs)
            self.tadgan_model = None
            return self.tadgan_model

        if self.model_name in {'TADGAN_ver1', 'TADGAN_ver2', 'TADGAN_ver3'}:
            raise ValueError("TADGAN_ver1/2/3 are legacy names. Use TADGAN.")

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
            checkpoint = self._require_path(
                os.path.join(self.model_exp_dir, self.data_name, "checkpoints", "best_on_test", "generator.pt"),
                "CTGAN best_on_test generator.pt",
            )
            state = torch.load(checkpoint, map_location=generator._model._device, weights_only=True)
            generator._model._generator.load_state_dict(state)

            if save:
                synthetic_data = self._sample_ctgan_synthesizer(generator, seed, file_path)
            else:
                generator._set_random_state(seed)
                num_positive = self.data_loader.num_data // 2
                num_negative = self.data_loader.num_data - num_positive
                synthetic_data = generator.sample_from_conditions(
                    conditions=[
                        Condition(num_rows=num_positive, column_values={f'{self.data_loader.target}': 1}),
                        Condition(num_rows=num_negative, column_values={f'{self.data_loader.target}': 0}),
                    ])

            if save and verbose:
                print(f"✔️ {os.path.basename(file_path)} 생성 완료!")
            return synthetic_data

        if self.model_name == 'TabDDPM':
            from generation.TabDDPM.scripts import pipeline as tabddpm_pipeline
            checkpoints_dir = os.path.join(self.model_exp_dir, self.data_name, "checkpoints")
            model_path = self._require_path(
                os.path.join(checkpoints_dir, "model_best_on_test.pt"),
                "TabDDPM model_best_on_test.pt",
            )

            synthetic_data = tabddpm_pipeline.run_sample(
                data_name=self.data_name,
                data_dir=self.data_dir,
                exp_dir=self.model_exp_dir,
                save_dir=self.model_save_dir,
                sample_seed=seed,
                change_val=False,
                save=save,
                output_path=file_path if save else None,
                model_path=model_path,
                verbose=verbose,
            )
            if save and verbose:
                print(f"✔️ {os.path.basename(file_path)} 생성 완료!")
            return synthetic_data

        if self.model_name == 'STaSy':
            from generation.STaSy import run_lib as stasy_run_lib

            config = self._load_stasy_config()
            best_dir = self._require_path(
                os.path.join(self.model_exp_dir, self.data_name, "checkpoints", "best_on_test"),
                "STaSy best_on_test checkpoint directory",
            )
            self._set_config_value(config, "checkpoint_override_dir", best_dir)
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
            codi_flags.checkpoint_override_dir = self._require_path(
                os.path.join(self.model_exp_dir, self.data_name, "best_on_test"),
                "CoDi best_on_test checkpoint directory",
            )
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

            checkpoints_dir = os.path.join(self.model_exp_dir, self.data_name, "checkpoints")
            model_input = self._require_path(
                getattr(self, "autodiff_best_path", None) or os.path.join(checkpoints_dir, "best_on_test.pt"),
                "AutoDiff best_on_test.pt",
            )

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
                checkpoint_override_dir=self._require_path(
                    os.path.join(self.exp_dir, "TTGAN", self.data_name, "generators", "best_on_test"),
                    "TTGAN best_on_test generator directory",
                ),
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
            from generation.TADGAN.scripts import sample as tadgan_sample
            from generation.TADGAN.scripts import utils as tadgan_utils

            tadgan_args, run_dirs = self._build_tadgan_context()
            self.tadgan_run_dirs = run_dirs
            ckpt_path = self._require_path(
                self.tadgan_ckpt_path or tadgan_utils.build_checkpoint_path(
                    run_dirs, self.data_name, "TADGAN", "best_on_test"),
                "TADGAN best_on_test checkpoint",
            )

            _, synthetic_data = tadgan_sample.sample(
                tadgan_args,
                self.data_loader,
                run_dirs,
                ckpt_path=ckpt_path,
                model=None,
                session=None,
                return_frame=True,
                save=save,
                output_path=file_path if save else None,
                verbose=verbose,
            )
            if save and verbose:
                print(f"✔️ {os.path.basename(file_path)} 생성 완료!")
            return synthetic_data

        if self.model_name in {'TADGAN_ver1', 'TADGAN_ver2', 'TADGAN_ver3'}:
            raise ValueError("TADGAN_ver1/2/3 are legacy names. Use TADGAN.")

        raise ValueError(f"지원되지 않는 모델입니다: {self.model_name}")
