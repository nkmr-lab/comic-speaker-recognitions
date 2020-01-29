
import pandas as pd
import settings

manga109_parser = settings.manga109_parser
DATASET_NAME    = settings.DATASET_NAME
SCORE_TYPES     = ['neighbor']


def main():
    books = manga109_parser.books
    for i, book in enumerate(books):

        # データセットを取り出す
        dataset_path = Path(f'{DATASET_DIR}/{i+1:03}_{book}.csv')
        with open(dataset_path, 'r') as f:
            dataset = pd.read_csv(f)
        with open(f'{DATASET_DIR}/{i+1:03}_{book}_character.txt', 'r') as f:
            characters_target = f.read()
        characters_target = characters_target.split('\n')

        score_dir = f'data/scores/{i+1:03}_{book}'

        for score_type in SCORE_TYPES:
            with open(f'{score_dir}/{}_{DATASET_NAME}.csv', 'r') as f:
                score = pd.read_csv(f, index_col=0)

        


if __name__ == '__main__':
    main()
