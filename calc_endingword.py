
import MeCab
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import settings


DATASET_DIR     = 'data/dataset'
OUTPUT_DIR      = 'data/scores'
CALC_NAME       = Path(__file__).stem.lstrip('calc_')
CORPUS_PATH     = 'data/corpus/endword.txt'
manga109_parser = settings.manga109_parser
mecab           = MeCab.Tagger(f'-d {settings.MECAB_IPADIC_PATH}')
DATASET_NAME    = settings.DATASET_NAME


def main():
    # コーパスを取得
    with open(CORPUS_PATH, 'r') as f:
        end_words = f.read().split()

    books = manga109_parser.books
    pbar = tqdm(total=len(books))
    for i, book in enumerate(books):
        pbar.set_postfix(title=book)
        annotation, dataset, characters_target = init(i, book)

        # 1つのコミックの全てのテキストとidを取得
        texts = [{'text': t['#text'], 'id': t['@id']} for p in annotation['page'] for t in p['text']]
        endwords_incomic = set()
        for text in texts:
            # データセットの対象外のテキストは飛ばす
            if text['id'] not in dataset['annotation_id'].values:
                text['end_word'] = ''
                continue
            # この変数に代入するべき単語を探す
            end_word = ''
            for edw in end_words:
                if text['text'].endswith(edw):
                    end_word = edw
                    break
            text['end_word'] = end_word

            if end_word != '':
                endwords_incomic.add(end_word)

        # 事前に距離とフレームから求めたスコアを取り出す
        score_dir = f'data/scores/{i+1:03}_{book}'
        with open(f'{score_dir}/inframe_{DATASET_NAME}.csv', 'r') as f:
            inframe_score = pd.read_csv(f, index_col=0)
        with open(f'{score_dir}/neighbor_{DATASET_NAME}.csv', 'r') as f:
            neighbor_score = pd.read_csv(f, index_col=0)
        with open(f'{score_dir}/taildirection3_{DATASET_NAME}.csv', 'r') as f:
            taildirection_score = pd.read_csv(f, index_col=0)

        score_book = inframe_score + neighbor_score + taildirection_score
        scores_output_df = pd.DataFrame(0, index=dataset['annotation_id'].values, columns=characters_target)
        score_output_tmp = pd.DataFrame([], columns=characters_target)

        # 出現した語尾ごとに処理
        for edw_target in list(endwords_incomic):
            # 対象のテキスト
            texts_target = list(filter(lambda x: x['end_word'] == edw_target, texts))
            id_target = [t['id'] for t in texts_target]

            score_target = score_book[score_book.index.isin(id_target)]
            # スコアをもとに，発話者を推定する
            speaker_characters = score_target.idxmax(axis=1)
            # どの発話者が何回ずつ現れたか？
            speaker_count = speaker_characters.value_counts()
            speaker_count_per = speaker_count / len(speaker_characters)
            # 該当するテキストごとにスコアを挿入
            for text_id in id_target:
                score_output_tmp = score_output_tmp.append(speaker_count_per.rename(text_id))

        # もとの全体のデータフレームに結合する
        score_output_tmp = score_output_tmp.fillna(0)
        scores_output_df = scores_output_df.add(score_output_tmp, fill_value=0)

        with open(f'{OUTPUT_DIR}/{i+1:03}_{book}/{CALC_NAME}_{DATASET_NAME}.csv', 'w') as f:
            scores_output_df.to_csv(f)

        pbar.update(1)


def init(i, book):
    annotation = manga109_parser.get_annotation(book)
    dataset_path = Path(f'{DATASET_DIR}/{i+1:03}_{book}.csv')
    with open(dataset_path, 'r') as f:
        dataset = pd.read_csv(f)
    with open(f'{DATASET_DIR}/{i+1:03}_{book}_character.txt', 'r') as f:
        characters_target = f.read()
    characters_target = characters_target.split('\n')
    return annotation, dataset, characters_target


if __name__ == '__main__':
    main()
