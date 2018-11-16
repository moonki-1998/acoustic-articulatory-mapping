import numpy as np

from artkaldi.tasks.all.example_config import *

from artkaldi.utils.traindatautils import get_artinv_data
from artkaldi.utils.trainutils import eval_artinv_model

# CONFIGURATION #
task_cfg = mngu0_artinv_DNN

if 'mngu0' in task_cfg['dataset']:

    # -------------------------------- MNGU0 ----------------------------------- #
    testset = get_artinv_data(task_cfg=task_cfg, test=True)
    rmse, r = eval_artinv_model(task_cfg=task_cfg, testset=testset)

elif task_cfg['dataset'] == 'USCTIMIT':

    # -------------------------------------- USC-multi ------------------------ #
    testset = get_artinv_data(task_cfg=task_cfg, test=True)
    rmse, r = eval_artinv_model(task_cfg=task_cfg, testset=testset)

elif task_cfg['dataset'] in ['USCTIMITF1', 'USCTIMITF5', 'USCTIMITM1', 'USCTIMITM3']:

    # ----------------------------- USC-SD ----------------------------- #
    rmses = []
    rs = []
    for dset in ['F1', 'F5', 'M1', 'M3']:
        task_cfg['dataset'] = 'USCTIMIT' + dset
        testset = get_artinv_data(task_cfg=task_cfg, test=True)
        rmse, r = eval_artinv_model(task_cfg=task_cfg, testset=testset)
        rmses.append(rmse)
        rs.append(r)
    print('Test RMSE average: ' + str(np.mean(rmses)))
    print('Test Pearson average: ' + str(np.mean(rs)))

elif 'MOCHA' in task_cfg['dataset']:

    # ------------------------------- MOCHA-CV ----------------------------- #
    rmses = []
    rs = []
    for dset in range(1, 6):
        task_cfg['dataset'] = 'MOCHATIMIT' + str(dset)
        testset = get_artinv_data(task_cfg=task_cfg, test=True)
        rmse, r = eval_artinv_model(task_cfg=task_cfg, testset=testset)
        rmses.append(rmse)
        rs.append(r)
    print('Test RMSE average: ' + str(np.mean(rmses)))
    print('Test Pearson average: ' + str(np.mean(rs)))

else:
    raise NotImplementedError
