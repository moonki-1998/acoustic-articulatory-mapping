import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


MNGU0_ARTICULATORY_DIR = '/yourpath/MNGU0_txt/ema_basic'
USC_ARTICULATORY_DIR = '/yourpath/USCTIMIT/data/articulatory'
MOCHA_ARTICULATORY_DIR = '/yourpath/MOCHATIMIT/msak0_v1.1/ema'
MNGU0_LSF_DIR = '/yourpath/MNGU0_txt'

WORK_DIR = '/yourpath/keras_experiments'
KALDI_RECIPES_DIR = '/yourpath/kaldi/kaldi-recipes'

os.makedirs(WORK_DIR, exist_ok=True)
