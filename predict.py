
import pandas as pd
from pathlib import Path
import settings

DATASET_DIR     = 'data/dataset'
manga109_parser = settings.manga109_parser
DATASET_NAME    = settings.DATASET_NAME
# SCORE_TYPES     = ['neighbor']
# SCORE_TYPES     = ['neighbor', 'inframe']
SCORE_TYPES     = ['neighbor', 'inframe', 'firstperson']
# SCORE_TYPES     = ['neighbor', 'inframe', 'firstperson', 'bow']


def main():
    books = manga109_parser.books
    predict_all = pd.DataFrame([], columns=['book_id', 'annotation_id', 'character_id'])
    for i, book in enumerate(books):

        # データセットを取り出す
        dataset_path = Path(f'{DATASET_DIR}/{i+1:03}_{book}.csv')
        with open(dataset_path, 'r') as f:
            dataset = pd.read_csv(f)
        with open(f'{DATASET_DIR}/{i+1:03}_{book}_character.txt', 'r') as f:
            characters_target = f.read()
        characters_target = characters_target.split('\n')

        score_dir = f'data/scores/{i+1:03}_{book}'

        score_all = pd.DataFrame(0, index=dataset['annotation_id'].values, columns=characters_target)
        for score_type in SCORE_TYPES:
            with open(f'{score_dir}/{score_type}_{DATASET_NAME}.csv', 'r') as f:
                score = pd.read_csv(f, index_col=0)
            score_all = score_all + score

        if score_all.isnull().values.sum() != 0:
            print('err')

        speaker_predict = score_all.idxmax(axis=1)
        # print(speaker_predict)
        speaker_df = pd.DataFrame({'book_id': i + 1, 'annotation_id': speaker_predict.index, 'character_id': speaker_predict.values})
        # print(speaker_df)
        predict_all = pd.concat([predict_all, speaker_df])

    scoring_name = ''.join([t[0] for t in SCORE_TYPES])
    with open(f'data/predict/{DATASET_NAME}_{scoring_name}.csv', 'w') as f:
        predict_all.to_csv(f, index=None)


if __name__ == '__main__':
    main()
