
import pandas as pd
from sklearn.metrics import accuracy_score
import settings

DATASET_NAME    = settings.DATASET_NAME
SCORE_TYPES     = settings.SCORE_TYPES
scoring_name = ''.join([t[0] for t in SCORE_TYPES])


def main():
    with open(f'data/raw/datasets_{DATASET_NAME}.csv', 'r') as f:
        true = pd.read_csv(f, index_col='annotation_id')
    with open(f'data/predict/{DATASET_NAME}_{scoring_name}.csv', 'r') as f:
        pred = pd.read_csv(f, index_col='annotation_id')

    evaluate_df = pd.DataFrame({'true': true['character_id'], 'pred': pred['character_id']})
    print(DATASET_NAME, SCORE_TYPES)
    print(accuracy_score(y_true=evaluate_df['true'].values, y_pred=evaluate_df['pred'].values))


if __name__ == '__main__':
    main()
