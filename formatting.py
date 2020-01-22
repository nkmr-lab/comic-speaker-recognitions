# コマンドラインで受け取ったファイルをもとに正解データを作成

import pandas as pd
import os
import datetime
from pathlib import Path
from dotenv import load_dotenv
import manga109api
import settings
load_dotenv()


DATASET_NAME = settings.DATASET_NAME
DATASET_FILE = f'data/raw/datasets_{DATASET_NAME}.csv'
OUTPUT_DIR = 'data/dataset'
MANGA109_ROOT_DIR = os.environ['MANGA109_ROOT_DIR']
manga109parser = manga109api.Parser(MANGA109_ROOT_DIR)


def main():

    # 本のタイトル取得
    books = manga109parser.books

    # データセットの生データ取得
    with open(DATASET_FILE, 'r') as f:
        dataset_raw = pd.read_csv(f)

    # 本ごとにグループ分けしてデータセットとユニークなキャラクタIDを書き出す
    for book_id, data_book in dataset_raw.groupby('book_id'):
        data_book = data_book.sort_values(['page', 'annotation_id'])
        base_name = f'{book_id:03}_{books[book_id - 1]}'
        with open(f'{OUTPUT_DIR}/{base_name}.csv', 'w') as f:
            data_book[['page', 'annotation_id', 'character_id']].to_csv(f, index=None)
        with open(f'{OUTPUT_DIR}/{base_name}_character.txt', 'w') as f:
            f.write('\n'.join(data_book.character_id.unique()))

    # 整形前のデータセットへの絶対パスと日付を記録
    today_str = datetime.datetime.today().strftime('%Y/%m/%d %H:%M:%S')
    with open(f'{OUTPUT_DIR}/.version', 'w') as f:
        f.write(f'{DATASET_FILE}\n{today_str}')

    # スコア記録用のディレクトリが無かったら作る
    for i, book in enumerate(books):
        score_dir = Path(f'data/scores/{i+1:03}_{book}')
        if not score_dir.exists():
            score_dir.mkdir()


if __name__ == '__main__':
    main()
