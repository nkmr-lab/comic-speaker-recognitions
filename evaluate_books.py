

import pandas as pd
from sklearn.metrics import accuracy_score
# from tqdm import tqdm
import settings

DATASET_NAME    = settings.DATASET_NAME
manga109_parser = settings.manga109_parser
# SCORE_TYPES     = settings.SCORE_TYPES
# scoring_name = ''.join([t[0] for t in SCORE_TYPES])

SCORE_TYPES_LIST = [
    ['neighbor_nonface'],
    ['inframe'],
    ['taildirection3'],
    ['firstperson', 'endingword'],
    ['neighbor_nonface', 'inframe', 'taildirection3', 'firstperson', 'endingword'],
]


def main():

    header = ['book_id']
    for score_types in SCORE_TYPES_LIST:
        scoring_name = ''.join([t[0] for t in score_types])
        header.append(scoring_name)
    print(','.join(header))

    books = manga109_parser.books

    rows = [[str(i + 1)] for i in range(len(books))]

    for score_types in SCORE_TYPES_LIST:
        scoring_name = ''.join([t[0] for t in score_types])

        with open(f'data/raw/datasets_{DATASET_NAME}.csv', 'r') as f:
            true = pd.read_csv(f, index_col='annotation_id')
        with open(f'data/predict/{DATASET_NAME}_{scoring_name}.csv', 'r') as f:
            pred = pd.read_csv(f, index_col='annotation_id')

        evaluate_df = pd.DataFrame({'book_id': true['book_id'], 'true': true['character_id'], 'pred': pred['character_id']})

        for i, book in enumerate(books):
            evaluate_book_df = evaluate_df[evaluate_df.book_id == (i + 1)]
            accuracy = accuracy_score(y_true=evaluate_book_df['true'].values, y_pred=evaluate_book_df['pred'].values)
            rows[i].append(f'{accuracy:.3f}')

    for row in rows:
        print(','.join(row))


if __name__ == '__main__':
    main()
