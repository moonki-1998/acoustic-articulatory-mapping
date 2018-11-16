import os
from random import shuffle

from tqdm import tqdm

import kaldi_io
import numpy as np

from artkaldi.config import WORK_DIR
from artkaldi.data_io.augmentation import AugmentationDataset, ArtAsrAugmentationDataset
from artkaldi.data_io.dataset import Dataset
from artkaldi.utils.paths import get_recipe_dir
from artkaldi.utils.signals import make_same_frames, get_silence_indices, smooth_acoustic


class SpeechDataset(Dataset):
    def __init__(self, cfg, dset, smooth_first=False):
        self.cfg = cfg
        self.dset = 'dev' if dset == 'valid' else dset

        self.recipe_dir = get_recipe_dir(cfg=cfg)
        if smooth_first:
            ark_smoothed = self.save_smoothed_feats()
            self.feats, self.labels = self.read_custom_feats(custom_feats_ark=ark_smoothed)
        else:
            self.feats, self.labels = self.read_data()
        self.feats, self.labels, self.uttids, self.end_indexes = self.unify_data(self.feats, self.labels)

    def read_data(self):
        feat_path = os.path.join(self.recipe_dir, 'data', self.dset, 'feats.scp')
        if self.dset == 'train':
            label_path = os.path.join(self.recipe_dir, 'exp', self.cfg['ali_dir'])
        else:
            label_path = os.path.join(self.recipe_dir, 'exp', self.cfg['ali_dir'] + '_' + self.dset)
        feat_opts = "apply-cmvn --utt2spk=ark:{0} ark:{1} ark:- ark:- |". \
            format(os.path.join(self.recipe_dir, 'data', self.dset, 'utt2spk'),
                   os.path.join(self.recipe_dir, 'data', self.dset,
                                self.dset + '_cmvn_speaker.ark'))
        if self.cfg['feature_deltas']:
            feat_opts += " add-deltas --delta-order=2 ark:- ark:- |"
        if self.cfg['feature_context']:
            feat_opts += " splice-feats --left-context={0} --right-context={0} ark:- ark:- |". \
                format(str(self.cfg['feature_context']))
        label_opts = 'ali-to-pdf' if self.cfg['task'] == 'classification' else 'ali-to-phones --per-frame'

        feats = {k: m for k, m in kaldi_io.read_mat_ark(
            'ark:copy-feats scp:{} ark:- | {}'.format(feat_path, feat_opts))}
        lab = {k: v for k, v in kaldi_io.read_vec_int_ark(
            'gunzip -c {0}/ali*.gz | {1} {0}/final.mdl ark:- ark:-|'.format(label_path, label_opts))
               if k in feats}
        feats = {k: v for k, v in feats.items() if k in lab}

        return feats, lab

    def unify_data(self, feats, lab, optional_array=None):
        fea_conc = np.concatenate([v for k, v in sorted(feats.items())])
        lab_conc = np.concatenate([v for k, v in sorted(lab.items())])
        if optional_array:
            opt_conc = np.concatenate([v for k, v in sorted(optional_array.items())])
        names = [k for k, v in sorted(lab.items())]
        end_snt = 0
        end_indexes = []
        for k, v in tqdm(sorted(lab.items())):
            end_snt += v.shape[0]
            end_indexes.append(end_snt)

        if self.cfg['task'] == 'classification':
            lab = lab_conc.astype('int64')
        else:
            lab = lab_conc
        if optional_array:
            opt = opt_conc.astype('int64')
            return fea_conc, lab, opt, names, end_indexes
        return fea_conc, lab, names, end_indexes

    def get_data_dims(self, cfg):
        if cfg['network_type'] in ('CNN', 'RCNN'):
            height = 23 if cfg['feature_type'] == 'fbank' else 13
            feat_dim = (height, 3 * (cfg['feature_context'] * 2 + 1))
            if cfg['feature_type'] == 'lsf':
                height = 40
                feat_dim = (height, 10)
        else:
            feat_dim = self.feats[0].shape[0]
        if cfg['task'] == 'classification':
            return feat_dim, int(self.labels.max() - self.labels.min() + 1)
        elif cfg['task'] == 'regression':
            return feat_dim, self.labels[0].shape[0]

    def save_smoothed_feats(self):
        feat_path = os.path.join(self.recipe_dir, 'data', self.dset, 'feats.scp')
        if self.dset == 'train':
            label_path = os.path.join(self.recipe_dir, 'exp', self.cfg['ali_dir'])
        else:
            label_path = os.path.join(self.recipe_dir, 'exp', self.cfg['ali_dir'] + '_' + self.dset)
        feat_opts = "apply-cmvn --utt2spk=ark:{0} ark:{1} ark:- ark:- |". \
            format(os.path.join(self.recipe_dir, 'data', self.dset, 'utt2spk'),
                   os.path.join(self.recipe_dir, 'data', self.dset,
                                self.dset + '_cmvn_speaker.ark'))
        label_opts = 'ali-to-pdf' if self.cfg['task'] == 'classification' else 'ali-to-phones --per-frame'
        feats = {k: m for k, m in kaldi_io.read_mat_ark(
            'ark:copy-feats scp:{} ark:- | {}'.format(feat_path, feat_opts))}
        lab = {k: v for k, v in kaldi_io.read_vec_int_ark(
            'gunzip -c {0}/ali*.gz | {1} {0}/final.mdl ark:- ark:-|'.format(label_path, label_opts))
               if k in feats}
        feats = {k: v for k, v in feats.items() if k in lab}
        fname = os.path.join(self.recipe_dir, 'data', self.dset, 'smoothed.ark')
        f = kaldi_io.open_or_fd(fname, 'wb')
        for key in tqdm(feats):
            tmp = smooth_acoustic(feats[key])
            kaldi_io.write_mat(f, tmp, key)

        return fname

    def read_custom_feats(self, custom_feats_ark):
        feat_path = custom_feats_ark
        if self.dset == 'train':
            label_path = os.path.join(self.recipe_dir, 'exp', self.cfg['ali_dir'])
        else:
            label_path = os.path.join(self.recipe_dir, 'exp', self.cfg['ali_dir'] + '_' + self.dset)
        feat_opts = ''
        if self.cfg['feature_deltas']:
            feat_opts += " add-deltas --delta-order=2 ark:- ark:- |"
        if self.cfg['feature_context']:
            feat_opts += " splice-feats --left-context={0} --right-context={0} ark:- ark:- |". \
                format(str(self.cfg['feature_context']))
        label_opts = 'ali-to-pdf' if self.cfg['task'] == 'classification' else 'ali-to-phones --per-frame'

        feats = {k: m for k, m in kaldi_io.read_mat_ark(
            'ark:copy-feats ark:{} ark:- | {}'.format(feat_path, feat_opts))}
        lab = {k: v for k, v in kaldi_io.read_vec_int_ark(
            'gunzip -c {0}/ali*.gz | {1} {0}/final.mdl ark:- ark:-|'.format(label_path, label_opts))
               if k in feats}
        feats = {k: v for k, v in feats.items() if k in lab}

        return feats, lab

    def __getitem__(self, idx):
        return self.feats[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


class AugmentedSpeechDataset(SpeechDataset):
    """
        A dataset class where the speech features are augmented by an augmentation dataset (e.g. articulatory)
    """

    def __init__(self, cfg, dset, augmentation_data_dir, keep_dims):
        self.cfg = cfg
        self.dset = 'dev' if dset == 'valid' else dset
        self.recipe_dir = get_recipe_dir(cfg=cfg)
        # LOAD RAW SPEECH DATA #
        self.feats, self.labels = self.read_data()
        self.uttids = [key for key in self.feats]
        # READ AUGMENTATION DATA #
        self.augmentation_data_dir = augmentation_data_dir
        self.keep_dims = keep_dims
        augmentation_dataset = ArtAsrAugmentationDataset(cfg=cfg,
                                                         dset=dset,
                                                         datadir=self.augmentation_data_dir,
                                                         fileset=self.uttids,
                                                         keep_dims=keep_dims)
        # AUGMENT THE SPEECH DATA WITH THE AUGMENTATION DATA #
        self.feats = self.augment(self.feats, augmentation_dataset.data)
        # UNIFY THE UTTERANCES #
        self.feats, self.labels, self.uttids, self.end_indexes = self.unify_data(self.feats, self.labels)

    @staticmethod
    def augment(feats, augmentation):
        tmp = {}
        for key in feats:
            # TRIM OR ADD FRAMES TO THE AUGMENTATION DATA
            f, a = make_same_frames(feats[key], augmentation[key])
            # CONCATENATE ALONG THE SECOND AXIS #
            tmp[key] = np.concatenate((f, a), axis=-1)

        return tmp


class ReconstructedSpeechDataset(SpeechDataset):
    def __init__(self, cfg, dset, augmentation_cfg):
        self.cfg = cfg
        self.dset = 'dev' if dset == 'valid' else dset
        self.recipe_dir = get_recipe_dir(cfg=cfg)
        # LOAD RAW SPEECH DATA #
        self.feats, self.labels = self.read_data()
        self.uttids = [key for key in self.feats]
        # READ AUGMENTATION DATA DICTIONARY#
        art_path = os.path.join(WORK_DIR, cfg['experiment_name'], cfg['dataset'], 'data_cache', 'articulatory')
        save_file_path = os.path.join(art_path,
                                      dset + '_' + augmentation_cfg['network_type'] + '_reconstructed_articulatory.ark')
        opts = ''
        if cfg['feature_context']:
            opts = "splice-feats --left-context={0} --right-context={0} ark:- ark:- |".format(
                str(cfg['feature_context']))
        augmentation_dict = {k: m for k, m in kaldi_io.read_mat_ark(
            'ark:copy-feats ark:{} ark:- | {}'.format(save_file_path, opts))}
        # AUGMENT THE SPEECH DATA WITH THE AUGMENTATION DATA #
        self.feats = self.augment(self.feats, augmentation_dict)
        # UNIFY THE UTTERANCES #
        self.feats, self.labels, self.uttids, self.end_indexes = self.unify_data(self.feats, self.labels)

    @staticmethod
    def augment(feats, augmentation):
        tmp = {}
        for key in feats:
            # CONCATENATE ALONG THE SECOND AXIS #
            tmp[key] = np.concatenate((feats[key], augmentation[key]), axis=-1)

        return tmp


class InverseMappingDataset(SpeechDataset):
    """
        A dataset class where the phone-labels are substituted with another feature set (e.g. articulatory)
    """

    def __init__(self, cfg, dset, augmentation_data_dir, keep_dims, cut_silence=True, smooth_first=True):
        self.cfg = cfg
        self.dset = 'dev' if dset == 'valid' else dset
        self.recipe_dir = get_recipe_dir(cfg=cfg)
        # LOAD RAW SPEECH DATA #
        if smooth_first:
            ark_smoothed = self.save_smoothed_feats()
            self.feats, self.labels = self.read_custom_feats(custom_feats_ark=ark_smoothed)
        else:
            self.feats, self.labels = self.read_data()
        self.uttids = [key for key in self.feats]
        # READ AUGMENTATION DATA #
        self.augmentation_data_dir = augmentation_data_dir
        self.keep_dims = keep_dims
        augmentation_dataset = AugmentationDataset(datadir=self.augmentation_data_dir,
                                                   fileset=self.uttids,
                                                   keep_dims=keep_dims)
        # PRODUCE THE NEW LABELS #
        self.cut_silence = cut_silence
        self.phone_labels = self.labels
        self.feats, self.labels, self.phone_labels = self.produce_labels(feats=self.feats,
                                                                         augmentation=augmentation_dataset.data,
                                                                         phones=self.phone_labels,
                                                                         cut_silence=self.cut_silence)
        # UNIFY THE UTTERANCES #
        self.feats, self.labels, self.phone_labels, self.uttids, self.end_indexes = self.unify_data(self.feats,
                                                                                                    self.labels,
                                                                                                    self.phone_labels)

    @staticmethod
    def produce_labels(feats, augmentation, phones, cut_silence=True):
        feat_dict = {}
        augm_dict = {}
        phone_dict = {}
        for key in tqdm(feats):
            # TRIM OR ADD FRAMES TO THE AUGMENTATION DATA
            f, a = make_same_frames(feats[key], augmentation[key])
            p = phones[key]
            # CUT SILENCE FRAMES #
            if cut_silence:
                silence_idx = get_silence_indices(phones=p)
                f = np.delete(f, silence_idx, 0)
                a = np.delete(a, silence_idx, 0)
                p = np.delete(p, silence_idx, 0)
            feat_dict[key] = f
            augm_dict[key] = a
            phone_dict[key] = p

        return feat_dict, augm_dict, phone_dict


class InverseMappingLosoDataset(SpeechDataset):
    """
        A dataset class where the phone-labels are substituted with another feature set (e.g. articulatory)
    """

    def __init__(self, cfg, speaker, augmentation_data_dir, keep_dims, cut_silence=True, smooth_first=True):
        self.cfg = cfg
        self.speaker = speaker
        self.recipe_dir = get_recipe_dir(cfg=cfg)

        self.dset = 'train'
        # LOAD RAW SPEECH DATA #
        if smooth_first:
            ark_smoothed = self.save_smoothed_feats()
            self.feats, self.labels = self.read_custom_feats(custom_feats_ark=ark_smoothed)
        else:
            self.feats, self.labels = self.read_data()
        self.uttids = [key for key in self.feats]
        # READ AUGMENTATION DATA #
        self.augmentation_data_dir = augmentation_data_dir
        self.keep_dims = keep_dims
        augmentation_dataset = AugmentationDataset(datadir=self.augmentation_data_dir,
                                                   fileset=self.uttids,
                                                   keep_dims=keep_dims)
        # PRODUCE THE NEW LABELS #
        self.cut_silence = cut_silence
        self.phone_labels = self.labels
        self.feats_tr, self.labels_tr, self.phone_labels_tr = self.produce_labels(feats=self.feats,
                                                                                  augmentation=augmentation_dataset.data,
                                                                                  phones=self.phone_labels,
                                                                                  cut_silence=self.cut_silence)
        self.dset = 'dev'
        # LOAD RAW SPEECH DATA #
        if smooth_first:
            ark_smoothed = self.save_smoothed_feats()
            self.feats, self.labels = self.read_custom_feats(custom_feats_ark=ark_smoothed)
        else:
            self.feats, self.labels = self.read_data()
        self.uttids = [key for key in self.feats]
        # READ AUGMENTATION DATA #
        self.augmentation_data_dir = augmentation_data_dir
        self.keep_dims = keep_dims
        augmentation_dataset = AugmentationDataset(datadir=self.augmentation_data_dir,
                                                   fileset=self.uttids,
                                                   keep_dims=keep_dims)
        # PRODUCE THE NEW LABELS #
        self.cut_silence = cut_silence
        self.phone_labels = self.labels
        self.feats_v, self.labels_v, self.phone_labels_v = self.produce_labels(feats=self.feats,
                                                                               augmentation=augmentation_dataset.data,
                                                                               phones=self.phone_labels,
                                                                               cut_silence=self.cut_silence)
        self.feats_tr.update(self.feats_v)
        self.labels_tr.update(self.labels_v)
        self.phone_labels_tr.update(self.phone_labels_v)

        self.dset = 'test'
        # LOAD RAW SPEECH DATA #
        if smooth_first:
            ark_smoothed = self.save_smoothed_feats()
            self.feats, self.labels = self.read_custom_feats(custom_feats_ark=ark_smoothed)
        else:
            self.feats, self.labels = self.read_data()
        self.uttids = [key for key in self.feats]
        # READ AUGMENTATION DATA #
        self.augmentation_data_dir = augmentation_data_dir
        self.keep_dims = keep_dims
        augmentation_dataset = AugmentationDataset(datadir=self.augmentation_data_dir,
                                                   fileset=self.uttids,
                                                   keep_dims=keep_dims)
        # PRODUCE THE NEW LABELS #
        self.cut_silence = cut_silence
        self.phone_labels = self.labels
        self.feats_t, self.labels_t, self.phone_labels_t = self.produce_labels(feats=self.feats,
                                                                               augmentation=augmentation_dataset.data,
                                                                               phones=self.phone_labels,
                                                                               cut_silence=self.cut_silence)
        self.feats_tr.update(self.feats_t)
        self.labels_tr.update(self.labels_t)
        self.phone_labels_tr.update(self.phone_labels_t)

        keys = [key for key in self.feats_tr if speaker not in key]
        shuffle(keys)
        train_keys = keys[:round(len(keys) * 0.9)]
        valid_keys = [key for key in keys if key not in train_keys]

        self.feats = {key: self.feats_tr[key] for key in train_keys}
        self.labels = {key: self.labels_tr[key] for key in train_keys}
        self.phone_labels = {key: self.phone_labels_tr[key] for key in train_keys}

        # UNIFY THE UTTERANCES #
        self.feats, self.labels, self.phone_labels, self.uttids, self.end_indexes = self.unify_data(self.feats,
                                                                                                    self.labels,
                                                                                                    self.phone_labels)
        self.feats_valid = {key: self.feats_tr[key] for key in valid_keys}
        self.labels_valid = {key: self.labels_tr[key] for key in valid_keys}
        self.phone_labels_valid = {key: self.phone_labels_tr[key] for key in valid_keys}

        # UNIFY THE UTTERANCES #
        self.feats_valid, self.labels_valid, self.phone_labels_valid, self.uttids_valid, self.end_indexes_valid = self.unify_data(
            self.feats_valid,
            self.labels_valid,
            self.phone_labels_valid)

        self.feats_test = {key: self.feats_tr[key] for key in self.feats_tr if speaker in key}
        self.labels_test = {key: self.labels_tr[key] for key in self.labels_tr if speaker in key}
        self.phone_labels_test = {key: self.phone_labels_tr[key] for key in self.phone_labels_tr if speaker in key}

        # UNIFY THE UTTERANCES #
        self.feats_test, self.labels_test, self.phone_labels_test, self.uttids_test, self.end_indexes_test = self.unify_data(
            self.feats_test,
            self.labels_test,
            self.phone_labels_test)

    @staticmethod
    def produce_labels(feats, augmentation, phones, cut_silence=True):
        feat_dict = {}
        augm_dict = {}
        phone_dict = {}
        for key in tqdm(feats):
            # TRIM OR ADD FRAMES TO THE AUGMENTATION DATA
            f, a = make_same_frames(feats[key], augmentation[key])
            p = phones[key]
            # CUT SILENCE FRAMES #
            if cut_silence:
                silence_idx = get_silence_indices(phones=p)
                f = np.delete(f, silence_idx, 0)
                a = np.delete(a, silence_idx, 0)
                p = np.delete(p, silence_idx, 0)
            feat_dict[key] = f
            augm_dict[key] = a
            phone_dict[key] = p

        return feat_dict, augm_dict, phone_dict
