from keras import Sequential
from keras.layers import InputLayer, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Flatten, Dense, \
    Softmax, GaussianNoise
from keras.layers import Conv1D, MaxPooling1D
from keras import regularizers


class CNN(object):
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg['conv1d']:
            self.model = self.create_model_1d(cfg)
        else:
            self.model = self.create_model(cfg)

    @staticmethod
    def create_model(cfg):
        pool_size = (2, 2)  # size of pooling area for max pooling
        kernel_size = (8, 8)  # convolution kernel size
        input_shape = (cfg['input_dim'][0], cfg['input_dim'][1], 3)
        # Keras Model
        model = Sequential()
        model.add(Conv2D(64, kernel_size, kernel_initializer='he_normal', data_format='channels_last',
                         padding='same', input_shape=input_shape))
        model.add(BatchNormalization(axis=3))
        model.add(Activation(activation='elu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, kernel_size, kernel_initializer='he_normal', data_format='channels_last',
                         padding='same'))
        model.add(BatchNormalization(axis=3))
        model.add(Activation(activation='elu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, kernel_size, kernel_initializer='he_normal', data_format='channels_last',
                         padding='same'))
        model.add(BatchNormalization(axis=3))
        model.add(Activation(activation='elu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(1000))
        model.add(Activation(activation='elu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(1000))
        model.add(Activation(activation='elu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Dense(cfg['output_dim']))
        model.add(BatchNormalization())

        # Optional softmax layer
        if cfg['task'] == 'classification':
            model.add(Softmax())

        return model

    @staticmethod
    def create_model_1d(cfg):
        pool_size = cfg['pool_size']  # size of pooling area for max pooling
        kernel_size = cfg['kernel_size']  # convolution kernel size
        input_shape = (23, 3 * (2 * cfg['feature_context'] + 1))
        # Keras Model
        model = Sequential()

        model.add(InputLayer(batch_input_shape=(None, input_shape[0], input_shape[1]), name='input'))
        model.add(GaussianNoise(stddev=cfg['gaussian_noise']))

        for i in range(cfg['num_cnn_layers']):
            model.add(Conv1D(filters=cfg['filters'], kernel_size=kernel_size,
                             padding=cfg['padding'],
                             kernel_initializer='he_normal',
                             kernel_regularizer=regularizers.l2(cfg['weight_decay'])))
            model.add(Activation(activation=cfg['activation']))
            if cfg['batch_normalization']:
                model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=pool_size))
            model.add(Dropout(cfg['dropout']))

        model.add(Flatten())

        for i in range(cfg['num_ff_layers']):
            model.add(Dense(cfg['ff_layer_size']))
            model.add(Activation(cfg['activation']))
            if cfg['batch_normalization']:
                model.add(BatchNormalization())
            model.add(Dropout(cfg['dropout']))

        model.add(Dense(cfg['output_dim']))
        model.add(BatchNormalization())
        # Optional softmax layer
        if cfg['task'] == 'classification':
            model.add(Softmax())

        return model
