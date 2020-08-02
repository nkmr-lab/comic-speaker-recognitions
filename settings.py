
import os
import manga109api
from dotenv import load_dotenv

# manga109の設定
load_dotenv()
MANGA109_ROOT_DIR = os.environ['MANGA109_ROOT_DIR']
manga109_parser = manga109api.Parser(root_dir=MANGA109_ROOT_DIR)

# mecabのパス
MECAB_IPADIC_PATH = os.environ['MECAB_IPADIC_PATH']

# データセットの設定
DATASET_NAME = 'top5'

# 計算方法の設定
SCORE_TYPES = [
    'neighbor_nonface',
    'inframe',
    'taildirection3',
    'firstperson', 'endingword'
]

UPDATE = True
