import os
import pickle

import kaldi_io
import numpy as np
from keras.models import load_model

from sklearn.preprocessing import StandardScaler

from artkaldi.config import WORK_DIR
from artkaldi.data_io.kaldi import SpeechDataset, AugmentedSpeechDataset, InverseMappingDataset, \
    ReconstructedSpeechDataset
from artkaldi.utils.image import prepare_cnn_input
from artkaldi.utils.metrics import rmse_avg
from artkaldi.utils.paths import get_kaldi_cache_name, get_recipe_dir, get_task_dirs, get_data_description_string
from artkaldi.utils.signals import smooth_acoustic
from artkaldi.utils.training import lstm_regression_results, get_bppt_dim


def load_kaldi_dataset(cfg, dset, smooth_first=False, standardize=False):
    save_dir = os.path.join(WORK_DIR, cfg['experiment_name'], cfg['dataset'], 'data_cache')
    os.makedirs(save_dir, exist_ok=True)
    save_name = os.path.join(save_dir, get_kaldi_cache_name(cfg, dset))
    scaler_name = os.path.join(save_dir, get_kaldi_cache_name(cfg, dset=None).split('.')[0] + '_scaler.pkl')
    if smooth_first:
        save_name += '_smoothed_acoustic'
        scaler_name += '_smoothed_acoustic'

    if not os.path.isfile(save_name):
        print('Saving to cache...')
        dataset = SpeechDataset(cfg=cfg, dset=dset, smooth_first=smooth_first)
        if dset == 'train':
            # Standardizer
            scaler = StandardScaler()
            scaler.fit(dataset.feats)
            with open(scaler_name, 'wb') as f:
                pickle.dump(scaler, f)
        # Save the dataset
        with open(save_name, 'wb') as f:
            pickle.dump(dataset, f)
    else:
        print('Cache exists...Reading...')
        with open(save_name, 'rb') as f:
            dataset = pickle.load(f)

    if standardize and not os.path.isfile(scaler_name):
        print('Standardize selected but there is no scaler!')
        raise FileNotFoundError

    if standardize:
        with open(scaler_name, 'rb') as f:
            scaler = pickle.load(f)
        dataset.feats = scaler.transform(dataset.feats)

    return dataset


def load_augmented_dataset(cfg, dset, augmentation_data_dir, keep_dims=None, standardize=False):
    save_dir = os.path.join(WORK_DIR, cfg['experiment_name'], cfg['dataset'], 'data_cache')
    os.makedirs(save_dir, exist_ok=True)
    save_name = os.path.join(save_dir, get_kaldi_cache_name(cfg, dset))
    scaler_name = os.path.join(save_dir, get_kaldi_cache_name(cfg, dset=None).split('.')[0] + '_scaler.pkl')

    dataset = AugmentedSpeechDataset(cfg=cfg, dset=dset,
                                     augmentation_data_dir=augmentation_data_dir, keep_dims=keep_dims)
    if not os.path.isfile(scaler_name):
        print('Saving scaler to cache...')
        if dset == 'train':
            # Standardizer
            scaler = StandardScaler(copy=False)
            mean_ = dataset.feats.mean(0)
            std_ = dataset.feats.std(0)
            scaler.mean_ = mean_
            scaler.scale_ = std_
            with open(scaler_name, 'wb') as f:
                pickle.dump(scaler, f)

    if standardize and not os.path.isfile(scaler_name):
        print('Standardize selected but there is no scaler!')
        raise FileNotFoundError

    if standardize:
        with open(scaler_name, 'rb') as f:
            scaler = pickle.load(f)
        dataset.feats = scaler.transform(dataset.feats)

    return dataset


def load_inverse_mapping_dataset(cfg, dset, augmentation_data_dir, keep_dims=None, cut_silence=True,
                                 smooth_first=True, standardize=False):
    save_dir = os.path.join(WORK_DIR, cfg['experiment_name'], cfg['dataset'], 'data_cache')
    os.makedirs(save_dir, exist_ok=True)
    save_name = os.path.join(save_dir, get_kaldi_cache_name(cfg, dset))
    feat_scaler_name = os.path.join(save_dir, get_kaldi_cache_name(cfg, dset=None).split('.')[0] + '_scaler.pkl')
    lab_scaler_name = os.path.join(save_dir, get_kaldi_cache_name(cfg, dset=None).split('.')[0] + '_label_scaler.pkl')

    if not os.path.isfile(save_name):
        print('Saving to cache...')
        dataset = InverseMappingDataset(cfg=cfg, dset=dset, augmentation_data_dir=augmentation_data_dir,
                                        keep_dims=keep_dims, cut_silence=cut_silence, smooth_first=smooth_first)
        if dset == 'train':
            # Standardizer
            feat_scaler = StandardScaler()
            feat_scaler.fit(dataset.feats)
            lab_scaler = StandardScaler()
            lab_scaler.fit(dataset.labels)
            with open(feat_scaler_name, 'wb') as f:
                pickle.dump(feat_scaler, f)
            with open(lab_scaler_name, 'wb') as f:
                pickle.dump(lab_scaler, f)
        # Save the dataset
        with open(save_name, 'wb') as f:
            pickle.dump(dataset, f)
    else:
        print('Cache exists...Reading...')
        with open(save_name, 'rb') as f:
            dataset = pickle.load(f)

    if standardize and not os.path.isfile(feat_scaler_name):
        print('Standardize selected but there is no scaler!')
        raise FileNotFoundError

    if standardize:
        with open(feat_scaler_name, 'rb') as f:
            feat_scaler = pickle.load(f)
        with open(lab_scaler_name, 'rb') as f:
            lab_scaler = pickle.load(f)
        dataset.feats = feat_scaler.transform(dataset.feats)
        dataset.labels = lab_scaler.transform(dataset.labels)

    return dataset


def load_loso_dataset(cfg, test=False):
    data_cache = os.path.join(WORK_DIR, cfg['experiment_name'], cfg['dataset'], 'data_cache')
    model_string = get_data_description_string(cfg, dset=None)
    save_name = os.path.join(data_cache, model_string + '_' + cfg['speaker'] + '.pkl')
    save_name = save_name.replace('_loso', '')
    with open(save_name, 'rb') as f:
        dataset = pickle.load(f)
    if test:
        return dataset.feats_test, dataset.labels_test
    else:
        return dataset


def load_counts(cfg):
    class_counts_file = os.path.join(get_recipe_dir(cfg), 'exp/{}/ali_train_pdf.counts'.format(cfg['ali_dir']))
    with open(class_counts_file) as f:
        row = f.readline().strip().strip('[]').strip()
        counts = np.array([np.float32(v) for v in row.split()])
    return counts


def save_regression_predictions_to_pickle(cfg, dataset, model, file, unscale=True):
    save_dir = os.path.join(WORK_DIR, cfg['experiment_name'], cfg['dataset'], 'data_cache')
    lab_scaler_name = os.path.join(save_dir, get_kaldi_cache_name(cfg, dset=None).split('.')[0] + '_label_scaler.pkl')
    if unscale:
        with open(lab_scaler_name, 'rb') as f:
            lab_scaler = pickle.load(f)
    tosave = {}
    start_index = 0
    dataset.end_indexes[-1] += 1
    for i, name in enumerate(dataset.uttids):
        tmp = model.predict(x=dataset.feats[start_index:dataset.end_indexes[i]])
        if unscale:
            tosave[name] = lab_scaler.inverse_transform(tmp)
        else:
            tosave[name] = tmp
        start_index = dataset.end_indexes[i]
    with open(file, 'wb') as f:
        pickle.dump(tosave, f)


def save_LSTM_regression_predictions_to_pickle(cfg, dataset, model, file, unscale=True):
    save_dir = os.path.join(WORK_DIR, cfg['experiment_name'], cfg['dataset'], 'data_cache')
    lab_scaler_name = os.path.join(save_dir, get_kaldi_cache_name(cfg, dset=None).split('.')[0] + '_label_scaler.pkl')
    if unscale:
        with open(lab_scaler_name, 'rb') as f:
            lab_scaler = pickle.load(f)
    # LSTM predictions #
    pred = lstm_regression_results(cfg=cfg, feats=dataset.feats, model=model)
    tosave = {}
    start_index = 0
    dataset.end_indexes[-1] += 1
    for i, name in enumerate(dataset.uttids):
        tmp = pred[start_index:dataset.end_indexes[i]]
        if unscale:
            tosave[name] = lab_scaler.inverse_transform(tmp)
        else:
            tosave[name] = tmp
        start_index = dataset.end_indexes[i]
    with open(file, 'wb') as f:
        pickle.dump(tosave, f)


def prepare_reconstructed_articulatory(cfg, dset, augmentation_cfg):
    # LOAD SPEECH DATASET
    cfg['experiment_name'] = 'asr_per'
    dataset = load_kaldi_dataset(cfg, dset, smooth_first=True, standardize=True)
    cfg['experiment_name'] = 'articulatory_asr'
    task_dirs = get_task_dirs(cfg=augmentation_cfg)

    # LOAD BEST MODEL AND PREDICT ARTICULATORY
    print('Predicting articulatory...')
    model = load_model(task_dirs['model'], custom_objects={'rmse_avg': rmse_avg})
    if augmentation_cfg['network_type'] == 'DNN':
        pred = model.predict(x=dataset.feats)
    elif augmentation_cfg['network_type'] == 'LSTM':
        pred = lstm_regression_results(cfg=augmentation_cfg, feats=dataset.feats, model=model)
    elif augmentation_cfg['network_type'] == 'CNN':
        feats = prepare_cnn_input(cfg=augmentation_cfg, feats=dataset.feats)
        pred = model.predict(x=feats)
    else:
        raise NotImplementedError

    # DENORMALIZE RECONSTRUCTED ARTICULATORY
    save_dir = os.path.join(WORK_DIR, augmentation_cfg['experiment_name'], augmentation_cfg['dataset'], 'data_cache')
    lab_scaler_name = os.path.join(save_dir, get_kaldi_cache_name(augmentation_cfg, dset=None).split('.')[
        0] + '_label_scaler.pkl')
    with open(lab_scaler_name, 'rb') as f:
        lab_scaler = pickle.load(f)
    rec = lab_scaler.inverse_transform(pred)

    # SAVE RECONSTRUCTED ARTICULATORY TO PICKLE BY UTTERANCE
    print('Saving articulatory to kaldi_format...')
    art_path = os.path.join(WORK_DIR, cfg['experiment_name'], cfg['dataset'], 'data_cache', 'articulatory')
    save_file_path = os.path.join(art_path,
                                  dset + '_' + augmentation_cfg['network_type'] + '_reconstructed_articulatory.ark')
    if not os.path.exists(art_path):
        os.makedirs(art_path, exist_ok=True)
    save_file = kaldi_io.open_or_fd(save_file_path, 'wb')
    start_index = 0
    dataset.end_indexes[-1] += 1
    for i, name in enumerate(dataset.uttids):
        out = rec[start_index:dataset.end_indexes[i]]
        start_index = dataset.end_indexes[i]
        kaldi_io.write_mat(save_file, out, dataset.uttids[i])


def load_reconstructed_dataset(cfg, dset, augmentation_cfg, standardize=False):
    save_dir = os.path.join(WORK_DIR, cfg['experiment_name'], cfg['dataset'], 'data_cache')
    os.makedirs(save_dir, exist_ok=True)
    scaler_name = os.path.join(save_dir,
                               get_kaldi_cache_name(cfg, dset=None).split('.')[0] + 'reconstructed_scaler.pkl')

    dataset = ReconstructedSpeechDataset(cfg=cfg, dset=dset, augmentation_cfg=augmentation_cfg)
    if dset == 'train':
        print('Saving scaler to cache...')
        # Standardizer
        scaler = StandardScaler(copy=False)
        mean_ = dataset.feats.mean(0)
        std_ = dataset.feats.std(0)
        scaler.mean_ = mean_
        scaler.scale_ = std_
        with open(scaler_name, 'wb') as f:
            pickle.dump(scaler, f)

    if standardize and not os.path.isfile(scaler_name):
        print('Standardize selected but there is no scaler!')
        raise FileNotFoundError

    if standardize:
        with open(scaler_name, 'rb') as f:
            scaler = pickle.load(f)
        dataset.feats = scaler.transform(dataset.feats)

    return dataset


def load_augment_from_pickle(speech_cfg, augmentation_cfg, dset, standardize=False):
    dataset = load_kaldi_dataset(cfg=speech_cfg, dset=dset, standardize=standardize)
    task_dirs = get_task_dirs(cfg=augmentation_cfg)
    best_model = load_model(task_dirs['model'], custom_objects={'rmse_avg': rmse_avg})

    if augmentation_cfg['network_type'] == 'LSTM':
        timestep = augmentation_cfg['timestep']
        dim = get_bppt_dim(len(dataset.feats), timestep=timestep)
        feats = dataset.feats[:dim, :].reshape((-1, timestep, augmentation_cfg['input_dim']))
        pred = best_model.predict(x=smooth_acoustic(feats))
        pred = pred.reshape((-1, 12))
    elif augmentation_cfg['network_type'] == 'CNN':
        feats = prepare_cnn_input(cfg=augmentation_cfg, feats=dataset.feats)
        pred = best_model.predict(x=smooth_acoustic(feats))
    elif augmentation_cfg['network_type'] == 'DNN':
        feats = dataset.feats
        pred = best_model.predict(x=smooth_acoustic(feats))
    else:
        raise NotImplementedError
    dataset.feats = np.concatenate((dataset.feats, pred), axis=1)

    return dataset
