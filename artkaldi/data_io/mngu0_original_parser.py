import os
import pickle

import numpy as np

from artkaldi.config import MNGU0_LSF_DIR, WORK_DIR
from artkaldi.data_io.dataset import Dataset
from artkaldi.utils.paths import get_kaldi_cache_name


class LsfDataset(Dataset):
    def __init__(self, cfg, dset):
        self.cfg = cfg
        self.readdir = os.path.join(MNGU0_LSF_DIR)

        save_dir = os.path.join(WORK_DIR, cfg['experiment_name'], cfg['dataset'], 'data_cache')
        os.makedirs(save_dir, exist_ok=True)
        save_name = os.path.join(save_dir, get_kaldi_cache_name(cfg, dset))

        if not os.path.isfile(save_name):
            self.feats = np.loadtxt(os.path.join(self.readdir, 'lsf_norm', dset, dset + '.txt'))
            self.labels = np.loadtxt(os.path.join(self.readdir, 'ema_norm_traj', dset, dset + '.txt'))
            self.feats = 4 * self.feats
            self.labels = 4 * self.labels
            # Save the dataset
            with open(save_name, 'wb') as f:
                pickle.dump((self.feats, self.labels), f)
        else:
            print('Cache exists...Reading...')
            with open(save_name, 'rb') as f:
                self.feats, self.labels = pickle.load(f)

    def return_dset(self):
        return self.feats, self.labels

    def __getitem__(self, idx):
        return self.feats[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def get_data_dims(self, cfg):
        if cfg['network_type'] == 'CNN':
            height = 23 if cfg['feature_type'] == 'fbank' else 13
            feat_dim = (height, cfg['feature_context'] * 2 + 1)
        else:
            feat_dim = self.feats[0].shape[0]
        if cfg['task'] == 'classification':
            return feat_dim, int(self.labels.max() - self.labels.min() + 1)
        elif cfg['task'] == 'regression':
            return feat_dim, self.labels[0].shape[0]
