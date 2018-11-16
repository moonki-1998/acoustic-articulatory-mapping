from keras import Sequential
from keras.layers import Softmax, Bidirectional, LSTM, TimeDistributed, Dense, BatchNormalization, CuDNNLSTM, Dropout, \
    GaussianNoise, regularizers


class LSTMart(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = self.create_model(cfg)

    @staticmethod
    def create_model(cfg):
        # Keras Model
        seq_length = cfg['timestep']
        model = Sequential()
        model.add(TimeDistributed(GaussianNoise(cfg['gaussian_noise']), input_shape=(seq_length, cfg['input_dim'])))
        if cfg['bidirectional']:
            model.add(Bidirectional(
                CuDNNLSTM(cfg['layer_size'], return_sequences=True,
                          kernel_regularizer=regularizers.l2(cfg['weight_decay'])), merge_mode='ave'))
            if cfg['batch_normalization']:
                model.add(TimeDistributed(BatchNormalization()))
            model.add(TimeDistributed(Dropout(rate=cfg['dropout'])))
            for i in range(cfg['num_layers'] - 1):
                model.add(Bidirectional(
                    CuDNNLSTM(cfg['layer_size'], return_sequences=True,
                              kernel_regularizer=regularizers.l2(cfg['weight_decay'])), merge_mode='ave'))
                if cfg['batch_normalization']:
                    model.add(TimeDistributed(BatchNormalization()))
                model.add(TimeDistributed(Dropout(rate=cfg['dropout'])))
        else:
            model.add(
                CuDNNLSTM(cfg['layer_size'], return_sequences=True,
                          input_shape=(seq_length, cfg['input_dim'])))
            if cfg['batch_normalization']:
                model.add(TimeDistributed(BatchNormalization()))
            model.add(TimeDistributed(Dropout(rate=cfg['dropout'])))
            for i in range(cfg['num_layers'] - 1):
                model.add(
                    CuDNNLSTM(cfg['layer_size'], return_sequences=True))
                if cfg['batch_normalization']:
                    model.add(TimeDistributed(BatchNormalization()))
                model.add(TimeDistributed(Dropout(rate=cfg['dropout'])))

        model.add(TimeDistributed(Dense(cfg['output_dim'], kernel_regularizer=regularizers.l2(cfg['weight_decay']))))
        if cfg['batch_normalization']:
            model.add(TimeDistributed(BatchNormalization()))
        # Optional softmax layer
        if cfg['task'] == 'classification':
            model.add(Softmax())

        return model
