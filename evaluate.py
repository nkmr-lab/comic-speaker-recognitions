
import pandas as pd
from sklearn.metrics import accuracy_score


def main():
    with open('data/raw/datasets_top5.csv', 'r') as f:
        true = pd.read_csv(f, index_col='annotation_id')
    with open('data/predict/top5_nif.csv', 'r') as f:
        pred = pd.read_csv(f, index_col='annotation_id')

    evaluate_df = pd.DataFrame({'true': true['character_id'], 'pred': pred['character_id']})
    print(accuracy_score(y_true=evaluate_df['true'].values, y_pred=evaluate_df['pred'].values))


if __name__ == '__main__':
    main()
