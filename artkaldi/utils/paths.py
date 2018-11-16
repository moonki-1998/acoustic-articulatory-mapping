import os

from artkaldi.config import KALDI_RECIPES_DIR, WORK_DIR


def get_recipe_dir(cfg):
    r = os.path.join(KALDI_RECIPES_DIR, cfg['dataset'])
    if os.path.exists(r):
        return r
    else:
        raise NotADirectoryError


def get_out_dir(cfg):
    out_dir = os.path.join(KALDI_RECIPES_DIR, cfg['dataset'], 'exp', 'dnn')
    os.makedirs(out_dir, exist_ok=True)

    return out_dir


def get_task_dirs(cfg, reconstructed_augmentation=False):
    # Tensorboard logs directory and Data cache dir
    dirs = {'logs': os.path.join(WORK_DIR, 'logs'),
            'data_cache': os.path.join(WORK_DIR, cfg['experiment_name'], cfg['dataset'], 'data_cache')}
    # Model checpoints directory
    model_dir = os.path.join(WORK_DIR, cfg['experiment_name'], cfg['dataset'], 'models')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(dirs['logs'], exist_ok=True)
    os.system('rm -rf ' + dirs['logs'] + '/*')

    model_string = get_data_description_string(cfg, dset=None)
    if cfg['network_type'] == 'DNN':
        model_string += '_model_' + str([cfg['layer_size'] for _ in range(cfg['num_hidden_layers'])])
    elif cfg['network_type'] == 'LSTM':
        model_string += '_model_' + str(cfg['num_layers']) + '_' + str(cfg['layer_size']) + '_LSTM'
        if cfg['bidirectional']:
            model_string += '_bi'
    elif cfg['network_type'] == 'CNN':
        model_string += '_model_' + 'CNN'  # TODO
        if cfg['conv1d']:
            model_string += '_1d'
    elif cfg['network_type'] == 'LRCN':
        model_string += '_model_' + 'LRCN'  # TODO
        if cfg['conv1d']:
            model_string += '_1d'
    else:
        raise NotImplementedError
    if cfg['experiment_name'] == 'articulatory_inversion_loso':
        model_string += '_' + cfg['speaker']
    dirs['model_string'] = model_string

    if not reconstructed_augmentation:
        dirs['model'] = os.path.join(model_dir, model_string + '.hdf5')
    else:
        dirs['model'] = os.path.join(model_dir, model_string + '_reconstructed_augmentation.hdf5')
    # Reconstructed augmentation features dir
    reconstructed_dir = os.path.join(dirs['data_cache'], 'reconstructed')
    os.makedirs(reconstructed_dir, exist_ok=True)
    for name in ['train', 'valid', 'test']:
        dirs['reconstructed_' + name] = os.path.join(reconstructed_dir,
                                                     model_string + '_reconstructed_{}.pkl'.format(name))

    return dirs


def get_kaldi_cache_name(cfg, dset=None):
    name = get_data_description_string(cfg, dset)
    name += '.pkl'

    return name


def get_data_description_string(cfg, dset=None):
    name = cfg['feature_type']
    if cfg['feature_deltas']:
        name += '_{}'.format('deltas')
    if cfg['feature_context']:
        name += '_{}_{}'.format('context', cfg['feature_context'])
    if cfg['experiment_name'] in ('asr_per', 'articulatory_asr'):
        if cfg['monophone_targets']:
            name += '_{}'.format('monophone_targets')
        else:
            name += '_{}'.format('context_targets')
    elif cfg['experiment_name'] == 'articulatory_inversion':
        if cfg['label_deltas']:
            name += '_{}'.format('label_deltas')
    elif cfg['experiment_name'] == 'articulatory_inversion_loso':
        if cfg['label_deltas']:
            name += '_{}'.format('label_deltas')
    else:
        raise NotImplementedError
    name += '_{}'.format(cfg['ali_dir'])
    if dset:
        name += '_' + dset

    return name
