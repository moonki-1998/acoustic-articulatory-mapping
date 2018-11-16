import numpy as np
import pandas as pd
import scipy.signal as signal


def interpolate_nans(array):
    df = pd.DataFrame(array)
    interpolated = df.interpolate().values

    return interpolated


def make_same_frames(target, subject):
    diff = len(target) - len(subject)
    last = abs(int(diff / 2))
    first = abs(abs(diff) - last)
    if diff == 0:
        return target, subject
    elif diff > 0:
        subj = np.pad(subject, ((first, last), (0, 0)), mode='edge')
        return target, subj
    else:
        subj = subject[first:-last, :]
        return target, subj


def get_silence_indices(phones):
    return np.argwhere(phones == 1)


def smooth_acoustic(feats):
    b, a = signal.butter(2, 0.25, btype='low')
    tmp = np.empty_like(feats)
    for i in range(feats.shape[1]):
        tmp[:, i] = signal.filtfilt(b, a, feats[:, i])
    return tmp
