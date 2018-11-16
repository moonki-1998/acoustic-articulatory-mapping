import numpy as np

from artkaldi.tasks.all.example_config import *

from artkaldi.utils.traindatautils import get_artinv_data
from artkaldi.utils.trainutils import train_model

# CONFIGURATION #
task_cfg = mngu0_artinv_DNN

if 'mngu0' in task_cfg['dataset']:

    # -------------------------------- MNGU0 ----------------------------------- #
    trainset, validset = get_artinv_data(task_cfg=task_cfg)
    train_model(task_cfg=task_cfg, trainset=trainset, validset=validset)

elif task_cfg['dataset'] == 'USCTIMIT':

    # -------------------------------------- USC-multi ------------------------ #
    trainset, validset = get_artinv_data(task_cfg=task_cfg)
    train_model(task_cfg=task_cfg, trainset=trainset, validset=validset)

elif task_cfg['dataset'] in ['USCTIMITF1', 'USCTIMITF5', 'USCTIMITM1', 'USCTIMITM3']:

    # ----------------------------- USC-SD ----------------------------- #
    for dset in ['F1', 'F5', 'M1', 'M3']:
        task_cfg['dataset'] = 'USCTIMIT' + dset
        trainset, validset = get_artinv_data(task_cfg=task_cfg)
        train_model(task_cfg=task_cfg, trainset=trainset, validset=validset)

elif 'MOCHA' in task_cfg['dataset']:

    # ------------------------------- MOCHA-CV ----------------------------- #
    for dset in range(1, 6):
        task_cfg['dataset'] = 'MOCHATIMIT' + str(dset)
        trainset, validset = get_artinv_data(task_cfg=task_cfg)
        train_model(task_cfg=task_cfg, trainset=trainset, validset=validset)

else:
    raise NotImplementedError
