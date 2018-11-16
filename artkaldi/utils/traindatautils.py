from artkaldi.config import MNGU0_ARTICULATORY_DIR, USC_ARTICULATORY_DIR, MOCHA_ARTICULATORY_DIR
from artkaldi.utils.datautils import load_inverse_mapping_dataset
from artkaldi.utils.image import prepare_cnn_input
from artkaldi.utils.training import get_bppt_dim


def get_artinv_data(task_cfg, test=False):
    if 'mngu0' in task_cfg['dataset']:
        articulatory_dir = MNGU0_ARTICULATORY_DIR
        keep_dims = (1, 2, 9, 10, 17, 18, 33, 34, 65, 66, 73, 74)
    elif 'USC' in task_cfg['dataset']:
        articulatory_dir = USC_ARTICULATORY_DIR
        keep_dims = None
    elif 'MOCHA' in task_cfg['dataset']:
        articulatory_dir = MOCHA_ARTICULATORY_DIR
        keep_dims = None
    else:
        raise NotImplementedError
    if not test:
        trainset = load_inverse_mapping_dataset(cfg=task_cfg, dset='train', augmentation_data_dir=articulatory_dir,
                                                keep_dims=keep_dims, cut_silence=True, smooth_first=True,
                                                standardize=True)
        validset = load_inverse_mapping_dataset(cfg=task_cfg, dset='valid', augmentation_data_dir=articulatory_dir,
                                                keep_dims=keep_dims, cut_silence=True, smooth_first=True,
                                                standardize=True)

        task_cfg['input_dim'], task_cfg['output_dim'] = trainset.get_data_dims(cfg=task_cfg)

        if task_cfg['network_type'] == 'LSTM':
            timestep = task_cfg['timestep']
            train_dim = get_bppt_dim(len(trainset.feats), timestep=timestep)
            valid_dim = get_bppt_dim(len(validset.feats), timestep=timestep)
            trainset.feats = trainset.feats[:train_dim, :].reshape((-1, timestep, task_cfg['input_dim']))
            trainset.labels = trainset.labels[:train_dim, :].reshape((-1, timestep, task_cfg['output_dim']))
            validset.feats = validset.feats[:valid_dim, :].reshape((-1, timestep, task_cfg['input_dim']))
            validset.labels = validset.labels[:valid_dim, :].reshape((-1, timestep, task_cfg['output_dim']))
        elif task_cfg['network_type'] == 'CNN':
            trainset.feats = prepare_cnn_input(cfg=task_cfg, feats=trainset.feats)
            validset.feats = prepare_cnn_input(cfg=task_cfg, feats=validset.feats)

        return trainset, validset

    else:
        testset = load_inverse_mapping_dataset(cfg=task_cfg, dset='test', augmentation_data_dir=articulatory_dir,
                                               keep_dims=keep_dims, cut_silence=True, smooth_first=True,
                                               standardize=True)
        task_cfg['input_dim'], task_cfg['output_dim'] = testset.get_data_dims(cfg=task_cfg)
        if task_cfg['network_type'] == 'LSTM':
            timestep = task_cfg['timestep']
            test_dim = get_bppt_dim(len(testset.feats), timestep=timestep)
            testset.feats = testset.feats[:test_dim, :].reshape((-1, timestep, task_cfg['input_dim']))
            testset.labels = testset.labels[:test_dim, :].reshape((-1, timestep, task_cfg['output_dim']))
        elif task_cfg['network_type'] == 'CNN':
            testset.feats = prepare_cnn_input(cfg=task_cfg, feats=testset.feats)

        return testset
