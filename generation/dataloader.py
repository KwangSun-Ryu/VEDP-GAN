'''
Create dataloaders required for synthetic data generation
'''

import os, sys, subprocess
from pathlib import Path
import json
import pandas as pd
from types import SimpleNamespace
from sdv.metadata import Metadata

class TabularDataLoader():
    def __init__(self, data_name, model_name, data_dir='./data', seed=42, prepare_data=True, verbose=True):
        self.data_name  = data_name  # data name to use
        self.model_name = model_name # model name to use
        self.data_dir   = data_dir   # path where datasets are stored
        self.seed       = seed       # seed value
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
        Create dataloaders for each model
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
            #### Create and save NumPy datasets ####
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
            data_loader = {'num_data': len(pd.read_csv(self.load_data_path())) } # add the data count
            if self.prepare_data:
                module_path  = 'generation.TTGAN.scripts.dataloader'
                command = [
                    sys.executable, '-m', module_path,
                    '--data-dir',  self.data_dir,
                    '--data-name', self.data_name,
                    '--seed',      str(self.seed) ]
                
                self._run_subprocess(command)
        
            return SimpleNamespace(**data_loader)

        if self.model_name == "VEDP-GAN":
            from generation.VEDP_GAN.scripts import dataloader as vedp_gan_dataloader

            vedp_gan_args = SimpleNamespace(
                data_name=self.data_name,
                data_dir=self.data_dir,
                model_name=self.model_name,
                batch_size=256,
                num_workers=2,
                bin_threshold=0.5,
                device="cuda" )

            return vedp_gan_dataloader.make_dataloader(vedp_gan_args)

    def load_cols_info_path(self):
        '''
        Load the JSON file path containing column information
        '''
        cols_info_path = os.path.join(self.data_dir, 'cols_info', f'{self.data_name}_metadata.json')
        assert os.path.exists(cols_info_path), f"{cols_info_path} does not exist."
        return cols_info_path

    def load_data_path(self):
        '''
        Load the data file path
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
