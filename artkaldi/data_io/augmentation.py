import os
import numpy as np
from tqdm import tqdm
import kaldi_io

from artkaldi.config import WORK_DIR
from artkaldi.data_io.dataset import Dataset
from artkaldi.utils.signals import interpolate_nans


class AugmentationDataset(Dataset):
    """
        Class that is used to read data that can augment a Speech Dataset
    """

    def __init__(self, datadir, fileset, keep_dims=None):
        self.datadir = datadir
        self.uttids = fileset
        self.keep_dims = keep_dims
        if keep_dims and not type(keep_dims) is tuple:
            print('keep_dims must be a tuple')
            raise ValueError
        self.filending = os.listdir(datadir)[0].split('.')[-1]
        self.filenames = [uttid + '.' + self.filending for uttid in self.uttids]
        self.data = self.read_augmentation_data()

    def read_augmentation_data(self):
        data = {}
        for uttid, filename in tqdm(zip(self.uttids, self.filenames)):
            # READ INITIAL DATA #
            raw = np.loadtxt(os.path.join(self.datadir, filename), dtype='float64')
            # KEEP SELECTED DIMENSIONS #
            if self.keep_dims:
                raw = raw[:, self.keep_dims]
            # INTERPOLATE NANS #
            raw = interpolate_nans(raw)
            # SAVE TO DICT ACCORDING TO THE UTTID #
            data[uttid] = raw

        return data

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


class ArtAsrAugmentationDataset(Dataset):
    """
        Class that is used to read data that can augment a Speech Dataset
    """

    def __init__(self, cfg, dset, datadir, fileset, keep_dims=None):
        art_path = os.path.join(WORK_DIR, cfg['experiment_name'], cfg['dataset'], 'data_cache', 'articulatory')
        save_file_path = os.path.join(art_path, dset + '_articulatory.ark')
        if not os.path.isfile(save_file_path):
            os.makedirs(art_path, exist_ok=True)
            save_file = kaldi_io.open_or_fd(save_file_path, 'wb')

            self.datadir = datadir
            self.uttids = fileset
            self.keep_dims = keep_dims
            if keep_dims and not type(keep_dims) is tuple:
                print('keep_dims must be a tuple')
                raise ValueError
            self.filending = os.listdir(datadir)[0].split('.')[-1]
            self.filenames = [uttid + '.' + self.filending for uttid in self.uttids]
            self.data = self.read_augmentation_data()
            for key, value in sorted(self.data.items()):
                kaldi_io.write_mat(save_file, value, key)
        opts = ''
        if cfg['feature_context']:
            opts = "splice-feats --left-context={0} --right-context={0} ark:- ark:- |".format(
                str(cfg['feature_context']))
        self.data = {k: m for k, m in kaldi_io.read_mat_ark(
            'ark:copy-feats ark:{} ark:- | {}'.format(save_file_path, opts))}

    def read_augmentation_data(self):
        data = {}
        for uttid, filename in tqdm(zip(self.uttids, self.filenames)):
            # READ INITIAL DATA #
            raw = np.loadtxt(os.path.join(self.datadir, filename), dtype='float64')
            # KEEP SELECTED DIMENSIONS #
            if self.keep_dims:
                raw = raw[:, self.keep_dims]
            # INTERPOLATE NANS #
            raw = interpolate_nans(raw)
            # SAVE TO DICT ACCORDING TO THE UTTID #
            data[uttid] = raw

        return data

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass