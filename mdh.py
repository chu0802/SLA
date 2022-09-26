import hashlib
from pathlib import Path
import pickle
from argparse import Namespace
from time import sleep
import re

def rmdir(path: Path):
    try:
        path.rmdir()
        if str(path) == 'log':
            return
        rmdir(path.parent)
    except:
        print(path)
        return

class GlobalHandler:
    def __init__(self, working_dir=Path('.')):
        self.working_dir = working_dir if isinstance(working_dir, Path) else Path(working_dir)
        self.model_dir = self.working_dir / 'model'
        self.log_dir = self.working_dir / 'log'
        self.feature_dir = self.working_dir / 'feature'
        self.table_path = self.working_dir / '.mdh/mdh_log.pk'

    def __repr__(self):
        return f"GlobalHandler(working_dir='{self.working_dir}')"

    def __str__(self):
        return f"GlobalHandler(working_dir='{self.working_dir}')"

    def hashing(self, cfg):
        return hashlib.md5(str(cfg).encode('utf-8')).hexdigest()

    def get_table(self):
        try:
            with open(self.table_path, 'rb') as f:
                table = pickle.load(f)
            return table
        except:
            return {}

    def dump_table(self, table):
        if len(table) > 0:
            with open(self.table_path, 'wb') as f:
                pickle.dump(table, f)

    def update(self, key, value):
        table = self.get_table()
        if key not in table:
            table[key] = value
            self.dump_table(table)

    def show(self):
        table = self.get_table()
        for k, v in table.items():
            print(f'{k}:')
            if isinstance(v, dict):
                for kk, vv in v.items():
                    print(f'\t{kk}: {vv}')

    def remove(self, key):
        table = self.get_table()
        if key in table:
            (self.model_dir / table[key]['model_path']).unlink(missing_ok=True)
            (self.feature_dir / table[key]['feature_path']).unlink(missing_ok=True)
            for d in (self.log_dir / table[key]['log_path']).iterdir():
                d.unlink()
            rmdir(self.log_dir / table[key]['log_path'])
            del table[key]
        self.dump_table(table)

        # if length of table equals 0, self.log_dir has already been removed.
        if len(table) == 0:
            self.model_dir.rmdir()
            self.feature_dir.rmdir()
            self.table_path.unlink()
            self.table_path.parent.rmdir()

    def remove_all(self):
        table = self.get_table()
        for k in table.keys():
            self.remove(k)
    def regSearch(self, reg):
        table = self.get_table()
        ret = list(filter(lambda x: re.search(reg, x['log_path']), table.values()))
        return [r['hashstr'] for r in ret]


    def getModelPath(self, hashstr):
        table = self.get_table()
        if hashstr in table:
            return self.model_dir / table[hashstr]['model_path']
    def getLogPath(self, hashstr):
        table = self.get_table()
        if hashstr in table:
            return self.log_dir / table[hashstr]['log_path']
    
    def getFeaturePath(self, hashstr):
        table = self.get_table()
        if hashstr in table:
            return self.feature_dir / table[hashstr]['feature_path']

class ModelHandler:
    def __init__(self, cfg, keys=None, gh=None):
        self.gh = GlobalHandler() if not gh else gh

        if isinstance(cfg, Namespace):
            cfg = cfg.__dict__

        self.cfg = {k: cfg[k] for k in sorted(keys)} if keys else dict(sorted(cfg.items(), key=lambda x: x[0]))
        self.hashstr = self.gh.hashing(self.cfg)

        self.model_path = f'{self.hashstr}.pt'
        self.feature_path = f'{self.hashstr}.npz'
        self.log_path = '/'.join([f'{k}:{v}' for k, v in self.cfg.items()])

        self.init()
        self.gh.update(self.hashstr, self.__dict__.copy())

    def init(self):
        self.getModelPath().parent.mkdir(parents=True, exist_ok=True)
        self.getFeaturePath().parent.mkdir(parents=True, exist_ok=True)
        self.getLogPath().mkdir(parents=True, exist_ok=True)
        self.gh.table_path.parent.mkdir(parents=True, exist_ok=True)

    def getHashStr(self):
        return self.hashstr

    def getModelPath(self):
        return self.gh.model_dir / self.model_path

    def getLogPath(self):
        return self.gh.log_dir / self.log_path
    
    def getFeaturePath(self):
        return self.gh.feature_dir / self.feature_path

    def remove(self):
        self.gh.remove(self.hashstr)
