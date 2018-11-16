from keras.models import Sequential
from keras.layers import InputLayer, Dense, BatchNormalization, Activation, Dropout, Softmax, GaussianNoise
from keras import regularizers


class DNN(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = self.create_model(cfg)

    @staticmethod
    def create_model(cfg):
        # Keras Model
        model = Sequential()
        # Input layer
        model.add(InputLayer(batch_input_shape=(None, cfg['input_dim']), name='input'))
        model.add(GaussianNoise(stddev=cfg['gaussian_noise']))
        # Hidden layers
        for i in range(cfg['num_hidden_layers'] - 1):
            model.add(Dense(units=cfg['layer_size'],
                            kernel_initializer='he_normal',
                            kernel_regularizer=regularizers.l2(cfg['weight_decay']),
                            name='hidden{}'.format(str(i))))
            model.add(Activation(activation=cfg['activation']))
            if cfg['batch_normalization']:
                model.add(BatchNormalization())
            model.add(Dropout(rate=cfg['dropout']))
        # Output layer
        model.add(Dense(units=cfg['output_dim'],
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(cfg['weight_decay']),
                        name='output'))
        if cfg['batch_normalization']:
            model.add(BatchNormalization())
        # Optional softmax layer
        if cfg['task'] == 'classification':
            model.add(Softmax())

        return model
