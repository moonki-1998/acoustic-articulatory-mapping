import numpy as np
from keras.models import load_model
from scipy.stats import pearsonr
from keras.callbacks import EarlyStopping, ModelCheckpoint

from artkaldi.utils.paths import get_task_dirs
from artkaldi.utils.training import get_model
from artkaldi.utils.signals import smooth_acoustic
from artkaldi.utils.metrics import rmse_avg
from sklearn.preprocessing import MinMaxScaler


def train_model(task_cfg, trainset, validset):
    # MODEL #
    print('Creating model with {} input dimension and {} output dimension.'.format(task_cfg['input_dim'],
                                                                                   task_cfg['output_dim']))
    model = get_model(cfg=task_cfg)

    # CALLBACKS #
    task_dirs = get_task_dirs(cfg=task_cfg)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    checkpointer = ModelCheckpoint(filepath=task_dirs['model'],
                                   monitor='val_loss', verbose=1, save_best_only=True)
    _callbacks = [early_stop, checkpointer]

    # TRAINING #
    model.fit(x=trainset.feats,
              y=trainset.labels,
              validation_data=(validset.feats, validset.labels),
              batch_size=task_cfg['batch_size'],
              epochs=task_cfg['epochs'],
              verbose=2,
              callbacks=_callbacks)


def eval_artinv_model(task_cfg, testset):
    # PRETRAINED MODEL #
    task_dirs = get_task_dirs(cfg=task_cfg)
    best_model = load_model(task_dirs['model'], custom_objects={'rmse_avg': rmse_avg})

    # EVALUATION #
    evaluation = best_model.evaluate(x=testset.feats,
                                     y=testset.labels,
                                     batch_size=len(testset))
    print('Test Loss: {}\nTest average RMSE: {}'.format(evaluation[0], evaluation[1]))
    pred = best_model.predict(x=testset.feats)
    if task_cfg['network_type'] == 'LSTM':
        testset.labels = testset.labels.reshape((-1, 12))
        pred = pred.reshape((-1, 12))
    p = []
    scaler = MinMaxScaler()
    lab = scaler.fit_transform(testset.labels)
    pr = scaler.fit_transform(pred)
    for i in range(pr.shape[1]):
        p.append(pearsonr(pr[:, i], lab[:, i])[0])
    print('Average Pearson: ' + str(np.array(p).mean()))

    return evaluation[1], np.array(p).mean()
