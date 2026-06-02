'''
합성 데이터 생성에 필요한 dataloader 만들기
'''

import os, sys, subprocess
from pathlib import Path
import json
import pandas as pd
from types import SimpleNamespace
from sdv.metadata import Metadata

class TabularDataLoader():
    def __init__(self, data_name, model_name, data_dir='./data', seed=42, prepare_data=True, verbose=True):
        self.data_name  = data_name  # 사용할 데이터 이름
        self.model_name = model_name # 사용할 모델 이름
        self.data_dir   = data_dir   # 데이터셋이 저장되어 있는 경로
        self.seed       = seed       # 시드값
        self.prepare_data = prepare_data
        self.verbose = bool(verbose)
        
        data_info = json.loads(Path(os.path.join(data_dir, 'datasets_info.json')).read_text(encoding='utf-8'))
        data_info = data_info[data_name]
        self.target = data_info['target']

    def _run_subprocess(self, command):
        run_kwargs = {'check': True}
        if not self.verbose:
            run_kwargs['stdout'] = subprocess.DEVNULL
            run_kwargs['stderr'] = subprocess.DEVNULL
        subprocess.run(command, **run_kwargs)

    def make_dataloader(self):
        '''
        각 모델에 맞게, dataloader 생성
        '''
        if self.model_name == 'CTGAN':
            metadata_path = self.load_cols_info_path()
            metadata = self._load_metadata(metadata_path)

            data_path = self.load_data_path()
            data = pd.read_csv(data_path)

            train_data = data.loc[data['split'] == 'train', :].drop(columns=['split'])
            test_data  = data.loc[data['split'] == 'test', :].drop(columns=['split'])

            data_loader = {'train_data': train_data, 
                           'test_data' : test_data, 
                           'metadata'  : metadata,
                           'target'    : self.target, 
                           'num_data':   len(data)    }

            return SimpleNamespace(**data_loader)
        
        if self.model_name == 'TabDDPM':
            #### numpy dataset 생성 및 저장 ####
            if not self.prepare_data:
                return None
            module_path = 'generation.TabDDPM.scripts.convert_my_data'
            command = [
                sys.executable, '-m', module_path,
                '--input-config', f'{os.path.join(self.data_dir, "datasets_info.json")}',
                '--metadata-dir', f'{os.path.join(self.data_dir, "cols_info")}',
                '--data-dir', f'{os.path.join(self.data_dir, "original_data")}',
                '--output-dir', f'{os.path.join(self.data_dir, f"{self.model_name}_data")}',
                '--datasets', self.data_name,
                '--val-ratio', '0.0' ]
            
            self._run_subprocess(command)
            
            return None
        
        if self.model_name == 'STaSy':
            if not self.prepare_data:
                return None
            module_path = 'generation.STaSy.scripts.convert_my_data'
            command = [
                sys.executable, '-m', module_path,
                '--input-config', f'{os.path.join(self.data_dir, "datasets_info.json")}',
                '--metadata-dir', f'{os.path.join(self.data_dir, "cols_info")}',
                '--data-dir', f'{os.path.join(self.data_dir, "original_data")}',
                '--output-dir', f'{os.path.join(self.data_dir, f"{self.model_name}_data")}',
                '--datasets', self.data_name ]
            
            self._run_subprocess(command)
            
            return None
        
        if self.model_name == 'CoDi':
            if not self.prepare_data:
                return None
            module_path = 'generation.CoDi.scripts.convert_my_data'
            command = [
                sys.executable, '-m', module_path,
                '--data-dir', f'{self.data_dir}',
                '--datasets', self.data_name ]
            
            self._run_subprocess(command)
            
            return None
        
        if self.model_name == 'AutoDiff':
            from generation.AutoDiff.scripts import pipeline as autodiff_pipeline

            auto_args = SimpleNamespace(
                data_name=self.data_name,
                model_name=self.model_name,
                data_dir=self.data_dir )

            return autodiff_pipeline._make_dataloader(auto_args)

        if self.model_name == 'TTGAN':
            data_loader = {'num_data': len(pd.read_csv(self.load_data_path())) } # 데이터 개수 추가
            if self.prepare_data:
                module_path  = 'generation.TTGAN.scripts.dataloader'
                command = [
                    sys.executable, '-m', module_path,
                    '--data-dir',  self.data_dir,
                    '--data-name', self.data_name,
                    '--seed',      str(self.seed) ]
                
                self._run_subprocess(command)
        
            return SimpleNamespace(**data_loader)

        if self.model_name == "TADGAN":
            from generation.TADGAN.scripts import dataloader as tadgan_dataloader

            tadgan_args = SimpleNamespace(
                data_name=self.data_name,
                data_dir=self.data_dir,
                model_name=self.model_name,
                batch_size=256,
                num_workers=2,
                bin_threshold=0.5,
                device="cuda" )

            return tadgan_dataloader.make_dataloader(tadgan_args)

        if self.model_name in {"TADGAN_ver1", "TADGAN_ver2", "TADGAN_ver3"}:
            raise ValueError("TADGAN_ver1/2/3 are legacy names. Use TADGAN.")

    def load_cols_info_path(self):
        '''
        열 정보를 담고 있는 json 파일 경로 불러오기
        '''
        cols_info_path = os.path.join(self.data_dir, 'cols_info', f'{self.data_name}_metadata.json')
        assert os.path.exists(cols_info_path), f"{cols_info_path}가 존재하지 않습니다."
        return cols_info_path

    def load_data_path(self):
        '''
        데이터 파일 경로 불러오기
        '''
        data_path = os.path.join(self.data_dir, 'original_data', f'{self.data_name}.csv')
        return data_path

    def _load_metadata(self, metadata_path):
        """Load SDV metadata using the unified ``Metadata`` API."""
        return Metadata.load_from_json(metadata_path, single_table_name=self.data_name)

if __name__ == '__main__':
    data_name = 'AFLD'
    model_name = 'TTGAN'
    data_loader = TabularDataLoader(data_name, model_name).make_dataloader()
    print(data_loader)
