
import pandas as pd
import settings

DATASET_NAME = settings.DATASET_NAME
manga109_parser = settings.manga109_parser


def main():
    books = manga109_parser.books

    is_tail_count = 0
    is_fe_count = 0

    for i, book in enumerate(books):
        with open(f'data/scores/{i+1:03}_{book}/taildirection3_{DATASET_NAME}.csv', 'r') as f:
            score_tail = pd.read_csv(f, index_col=0)
        is_tail_count = is_tail_count + (score_tail.sum(axis=1) != 0).sum()

        with open(f'data/scores/{i+1:03}_{book}/firstperson_{DATASET_NAME}.csv', 'r') as f:
            score_first = pd.read_csv(f, index_col=0)
        with open(f'data/scores/{i+1:03}_{book}/endingword_{DATASET_NAME}.csv', 'r') as f:
            score_end = pd.read_csv(f, index_col=0)
        score_fe = score_first + score_end
        is_fe_count = is_fe_count + (score_fe.sum(axis=1) != 0).sum()

    print('しっぽが含まれる', is_tail_count)
    print('一人称・語尾が含まれる', is_fe_count)


if __name__ == '__main__':
    main()
