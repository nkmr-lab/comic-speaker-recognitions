
import pandas as pd
from sklearn.metrics import accuracy_score
import settings

DATASET_NAME    = settings.DATASET_NAME

SCORE_TYPES_LIST = [
    ['neighbor_nonface'],
    ['inframe'],
    ['taildirection3'],
    ['firstperson', 'endingword'],
]

SCORE_TYPE_MULTI = ['neighbor_nonface', 'inframe', 'taildirection3', 'firstperson', 'endingword']


def main():

    print('type,accuracy,contribute')

    with open(f'data/raw/datasets_{DATASET_NAME}.csv', 'r') as f:
        true = pd.read_csv(f, index_col='annotation_id')

    with open(f'data/predict/{DATASET_NAME}_nitfe.csv', 'r') as f:
        pred_multi = pd.read_csv(f, index_col='annotation_id')

    evaluate_multi = pd.DataFrame({'true': true['character_id'], 'pred': pred_multi['character_id']})
    accuracy_multi = accuracy_score(y_true=evaluate_multi['true'].values, y_pred=evaluate_multi['pred'].values)

    for score_types in SCORE_TYPES_LIST:
        scoring_name = ''.join([t[0] for t in score_types])
        row = [scoring_name]

        with open(f'data/predict/{DATASET_NAME}_{scoring_name}.csv', 'r') as f:
            pred = pd.read_csv(f, index_col='annotation_id')

        evaluate_df = pd.DataFrame({'true': true['character_id'], 'pred': pred['character_id']})
        # print(DATASET_NAME, score_types)
        accuracy = accuracy_score(y_true=evaluate_df['true'].values, y_pred=evaluate_df['pred'].values)
        row.append(f'{accuracy:.3f}')

        score_types_comp = list(filter(lambda x: x not in score_types, SCORE_TYPE_MULTI))
        scoring_name = ''.join([t[0] for t in score_types_comp])
        with open(f'data/predict/{DATASET_NAME}_{scoring_name}.csv', 'r') as f:
            pred_comp = pd.read_csv(f, index_col='annotation_id')
        evaluate_df = pd.DataFrame({'true': true['character_id'], 'pred': pred_comp['character_id']})
        # print(DATASET_NAME, score_types)
        accuracy_comp = accuracy_score(y_true=evaluate_df['true'].values, y_pred=evaluate_df['pred'].values)
        row.append(f'{(accuracy_multi - accuracy_comp):.3f}')

        print(','.join(row))

    print(f'nitfe,{accuracy_multi:.3f},')


if __name__ == '__main__':
    main()
