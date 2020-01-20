
import pandas as pd
import os
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from dotenv import load_dotenv
load_dotenv()

DATASET_DIR = 'data/dataset'
SCORES_DIR = 'data/scores'
MANGA109_ROOT_DIR = os.environ['MANGA109_ROOT_DIR']


def main():
    # 本のタイトル取得
    with open(f'{MANGA109_ROOT_DIR}/books.txt', 'r') as f:
        books = f.read()
    books = books.split('\n')[:-1]

    print('book_id,title,accuracy,f1_macro')
    for i, book in enumerate(books):
        BASE_NAME = f'{i+1:03}_{book}'

        if Path(f'{DATASET_DIR}/{BASE_NAME}.csv').exists() is False:
            continue

        with open(f'{DATASET_DIR}/{BASE_NAME}.csv', 'r') as f:
            dataset = pd.read_csv(f, index_col=1)
        with open(f'{SCORES_DIR}/{BASE_NAME}_neighbor.csv', 'r') as f:
            scores_neighbor = pd.read_csv(f, index_col=0)
        dataset['predict'] = scores_neighbor.idxmax(axis=1)
        accuracy = accuracy_score(dataset['character_id'], dataset['predict'])
        precision = precision_score(dataset['character_id'], dataset['predict'], average='macro')

        print(f"{i+1},{book},{accuracy},{precision}")


if __name__ == '__main__':
    main()
