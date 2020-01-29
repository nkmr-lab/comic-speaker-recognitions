
import MeCab
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import settings


DATASET_DIR     = 'data/dataset'
OUTPUT_DIR      = 'data/scores'
CALC_NAME       = Path(__file__).stem.lstrip('calc_')
CORPUS_PATH     = 'data/corpus/fpp.csv'
manga109_parser = settings.manga109_parser
mecab           = MeCab.Tagger(f'-d {settings.MECAB_IPADIC_PATH}')
DATASET_NAME    = settings.DATASET_NAME


def main():
    # コーパスを取得
    with open(CORPUS_PATH, 'r') as f:
        fpp = f.read().split()[1:]
    fpp_katakana = [x.split(',')[0] for x in fpp]
    fpp_norm = [x.split(',')[1] for x in fpp]

    books = manga109_parser.books
    pbar = tqdm(total=len(books))
    for i, book in enumerate(books):
        pbar.set_postfix(title=book)
        annotation, dataset, characters_target = init(i, book)

        # 1つのコミックの全てのテキストとidを取得
        texts = [{'text': t['#text'], 'id': t['@id']} for p in annotation['page'] for t in p['text']]
        fpp_incomic = set()
        for text in texts:
            # データセットの対象外のテキストは飛ばす
            if text['id'] not in dataset['annotation_id'].values:
                text['fpp'] = ''
                continue
            # この変数に代入するべき単語を探す
            first_person_word = ''
            mecab.parse('')
            node = mecab.parseToNode(text['text'])
            while node:
                feature = node.feature.split(',')
                if feature[0] == '名詞':
                    if feature[1] == '代名詞':
                        word = feature[7]
                        if word in fpp_katakana:
                            # print(word)
                            first_person_word = word
                            break
                    elif feature[1] == '一般':
                        if len(feature) <= 7:
                            pass
                        elif feature[6] in fpp_norm:
                            num = fpp_norm.index(feature[6])
                            word = fpp_katakana[num]
                            first_person_word = word
                            break
                        elif feature[7] in fpp_katakana:
                            word = feature[7]
                            first_person_word = word
                            break
                node = node.next
            text['fpp'] = first_person_word
            if first_person_word != '':
                fpp_incomic.add(first_person_word)

        # 事前に距離とフレームから求めたスコアを取り出す
        score_dir = f'data/scores/{i+1:03}_{book}'
        with open(f'{score_dir}/inframe_{DATASET_NAME}.csv', 'r') as f:
            inframe_score = pd.read_csv(f, index_col=0)
        with open(f'{score_dir}/neighbor_{DATASET_NAME}.csv', 'r') as f:
            neighbor_score = pd.read_csv(f, index_col=0)

        score_book = inframe_score + neighbor_score
        scores_output_df = pd.DataFrame(0, index=dataset['annotation_id'].values, columns=characters_target)
        score_output_tmp = pd.DataFrame([], columns=characters_target)
        # scores_output_df = pd.DataFrame([], columns=characters_target)

        # 出現した一人称ごとに処理
        for fpp_target in list(fpp_incomic):
            # 対象のテキスト
            texts_target = list(filter(lambda x: x['fpp'] == fpp_target, texts))
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
