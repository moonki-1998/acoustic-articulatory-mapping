import keras.backend as K


def rmse_avg(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.sum(K.square(y_pred - y_true), axis=-1)))


def rmse_hard(y_true, y_pred):
    return K.sum(K.sqrt(K.mean(K.square(y_pred - y_true), axis=0)))
