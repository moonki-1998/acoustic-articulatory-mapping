import os
import pickle
import numpy as np

from keras.optimizers import SGD, Adam, RMSprop

from artkaldi.models.cnn import CNN
from artkaldi.models.dnn import DNN
from artkaldi.models.lrcn import LRCN
from artkaldi.models.rnn import LSTMart
from artkaldi.utils.metrics import rmse_avg
from artkaldi.utils.paths import get_task_dirs
from artkaldi.utils.signals import smooth_acoustic


def get_model(cfg):
    if cfg['network_type'] == 'DNN':
        model = DNN(cfg).model
    elif cfg['network_type'] == 'LSTM':
        model = LSTMart(cfg).model
    elif cfg['network_type'] == 'CNN':
        model = CNN(cfg).model
    elif cfg['network_type'] == 'LRCN':
        model = LRCN(cfg).model
    else:
        raise NotImplementedError

    opt = cfg['optimizer']
    if opt == 'sgd':
        optimizer = SGD(lr=cfg['learning_rate'], clipnorm=cfg['clipnorm'])
    elif opt == 'adam':
        optimizer = Adam(clipnorm=cfg['clipnorm'])
    elif opt == 'rmsprop':
        optimizer = RMSprop(clipnorm=cfg['clipnorm'])
    else:
        raise NotImplementedError

    # Compile model
    metrics = get_metrics(cfg=cfg)
    if cfg['task'] == 'regression':
        model.compile(optimizer=optimizer,
                      loss='mse',
                      metrics=metrics)
    elif cfg['task'] == 'classification':
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=metrics)

    return model


def get_metrics(cfg):
    if cfg['task'] == 'classification':
        metrics = ['accuracy']
    elif cfg['task'] == 'regression':
        metrics = [rmse_avg]
    else:
        raise NotImplementedError

    return metrics


def save_history(cfg, history):
    dirs = get_task_dirs(cfg)
    log_dir = dirs['logs']
    os.makedirs(log_dir, exist_ok=True)
    hist_string = 'history_' + dirs['model_string'] + '.pkl'
    hist_file = os.path.join(log_dir, hist_string)
    with open(hist_file, 'wb') as f:
        pickle.dump(history, f)


def get_bppt_dim(siz, timestep):
    dim = siz
    while dim % timestep != 0:
        dim -= 1

    return dim


def lstm_regression_results(cfg, feats, model):
    # LSTM PREPARATION #
    timestep = cfg['timestep']
    cfg['input_dim'] = feats.shape[1]
    train_dim = get_bppt_dim(len(feats), timestep=timestep)
    diff = feats.shape[0] - train_dim
    last_seq = feats[-timestep:, :]
    last_seq = last_seq.reshape((-1, timestep, cfg['input_dim']))
    feats = feats[:train_dim, :].reshape((-1, timestep, cfg['input_dim']))
    pred = model.predict(x=feats)
    pred = smooth_acoustic(pred.reshape((-1, 12)))
    last_pred = model.predict(x=last_seq)
    last_pred = smooth_acoustic(last_pred.reshape((-1, 12)))
    last_pred = last_pred[-diff:, :]

    return np.concatenate((pred, last_pred))
