import numpy as np


def prepare_cnn_input(cfg, feats):
    if cfg['feature_type'] == 'plp':
        dim = 13
    elif cfg['feature_type'] == 'fbank':
        dim = 23
    else:
        raise NotImplementedError
    total = cfg['feature_context'] * 2 + 1

    samples = feats.shape[0]
    if cfg['conv1d']:
        image = np.transpose(feats.reshape(-1, total * 3, dim), axes=(0, 2, 1))  # TODO Check if correct
    else:
        image = np.empty(shape=(samples, dim, total, 3))  # 3D image
        idx_static = np.concatenate(np.array([np.arange(i * 3 * dim, i * 3 * dim + dim) for i in range(total)]))
        idx_d = np.concatenate(np.array([np.arange(i * 3 * dim + dim, i * 3 * dim + 2 * dim) for i in range(total)]))
        idx_dd = np.concatenate(
            np.array([np.arange(i * 3 * dim + 2 * dim, i * 3 * dim + 3 * dim) for i in range(total)]))
        image[:, :, :, 0] = np.transpose(feats[:, idx_static].reshape((-1, total, dim)), axes=(0, 2, 1))
        image[:, :, :, 1] = np.transpose(feats[:, idx_d].reshape((-1, total, dim)), axes=(0, 2, 1))
        image[:, :, :, 2] = np.transpose(feats[:, idx_dd].reshape((-1, total, dim)), axes=(0, 2, 1))

    return image
