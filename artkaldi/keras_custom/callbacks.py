import os
import tensorflow as tf

import numpy as np
import keras.backend as K

from keras.callbacks import Callback, TensorBoard


class LRAnnealingCallback(Callback):
    def __init__(self, monitor='val_acc', improvement_threshold=0.001, halving_factor=0.5, verbose=0):
        super(LRAnnealingCallback, self).__init__()
        self.monitor = monitor
        self.improvement_threshold = improvement_threshold
        self.halving_factor = halving_factor
        self.verbose = verbose

        self.history = []

    def on_train_begin(self, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

    def on_epoch_end(self, epoch, logs=None):

        self.history.append(1 - logs[self.monitor])

        if epoch > 0:
            lr = float(K.get_value(self.model.optimizer.lr))

            lr, changed_flag = self.annealing(lr)

            if not isinstance(lr, (float, np.float32, np.float64)):
                raise ValueError('The output of the annealing function '
                                 'should be float.')
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0 and changed_flag:
                print('Epoch %05d: LearningRateAnnealing reducing learning '
                      'rate to %s.\n' % (epoch + 1, lr))

    def annealing(self, lr):
        flag = False
        test = (self.history[-2] - self.history[-1]) / self.history[-1]
        if test < self.improvement_threshold:
            lr *= self.halving_factor
            flag = True

        return lr, flag


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


class GradLogger(Callback):
    def __init__(self):
        super(GradLogger, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers:
            print('\t\tLayer: ')
            print(layer)
            for weight in layer.weights:
                grads = self.model.optimizer.get_gradients(self.model.total_loss, weight)
                # grads = [
                #     grad.values if is_indexed_slices(grad) else grad
                #     for grad in grads]
                print('\tWeight: ')
                print(weight)
                print('\tGrad: ')
                print(grads)


def is_indexed_slices(grad):
    return type(grad).__name__ == 'IndexedSlices'
