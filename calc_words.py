
import MeCab
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import settings

DATASET_DIR     = 'data/dataset'
OUTPUT_DIR      = 'data/scores'
CALC_NAME       = Path(__file__).stem.lstrip('calc_')
CORPUS_PATH     = 'data/corpus/fpp.csv'
manga109_parser = settings.manga109_parser
mecab           = MeCab.Tagger(f'-d {settings.MECAB_IPADIC_PATH}')
DATASET_NAME    = settings.DATASET_NAME

# コーパスを取得・一人称を省くために使う
with open(CORPUS_PATH, 'r') as f:
    fpp = f.read().split()[1:]
fpp_katakana = [x.split(',')[0] for x in fpp]
fpp_norm = [x.split(',')[1] for x in fpp]


def main():

    books = manga109_parser.books
    pbar = tqdm(total=len(books))
    for i, book in enumerate(books):
        annotation, dataset, characters_target = init(i, book)

        texts = [{'text': t['#text'], 'id': t['@id']} for p in annotation['page'] for t in p['text']]

        # 分かち書き
        for text in texts:
            text['wakachi'] = []
            mecab.parse('')
            node = mecab.parseToNode(text['text'])
            while node:
                feature = node.feature.split(',')
                if is_target_word(feature):
                    text['wakachi'].append(node.surface)

                node = node.next

        texts_target = list(filter(lambda x: x['id'] in dataset['annotation_id'].values, texts))
        texts_wakachi = [' '.join(t['wakachi']) for t in texts_target]
        texts_id      = [t['id'] for t in texts_target]

        max_features = None  # この数字を変えることで精度が上がるかも
        vectorizer = TfidfVectorizer(max_features=max_features)
        X = vectorizer.fit_transform(texts_wakachi)
        # print(vectorizer.get_feature_names(), len(vectorizer.get_feature_names()))

        n_clusters = len(characters_target)  # 必要であればここの数の倍率を変える
        # n_clusters = 5

        kmeans = KMeans(n_clusters=len(characters_target), random_state=0)
        kmeans.fit(X)
        labels = kmeans.labels_

        # 事前に距離とフレームから求めたスコアを取り出す
        score_dir = f'data/scores/{i+1:03}_{book}'
        with open(f'{score_dir}/inframe_{DATASET_NAME}.csv', 'r') as f:
            inframe_score = pd.read_csv(f, index_col=0)
        with open(f'{score_dir}/neighbor_{DATASET_NAME}.csv', 'r') as f:
            neighbor_score = pd.read_csv(f, index_col=0)
        score_book = inframe_score + neighbor_score

        scores_output_df = pd.DataFrame(0, index=dataset['annotation_id'].values, columns=characters_target)
        score_output_tmp = pd.DataFrame([], columns=characters_target)

        # print(kmeans.labels_.shape)
        for num in range(n_clusters):
            texts_id = np.array(texts_id)
            texts_id_inclass = texts_id[labels == num]
            # print(num, len(texts_id_inclass))
            # print(texts_id_inclass)
            score_target = score_book[score_book.index.isin(texts_id_inclass)]
            # print(score_target)
            speaker_characters = score_target.idxmax(axis=1)
            speaker_count = speaker_characters.value_counts()
            speaker_count_per = speaker_count / len(speaker_characters)
            # print(speaker_count_per)
            for id in texts_id_inclass:
                score_output_tmp = score_output_tmp.append(speaker_count_per.rename(id))

        score_output_tmp = score_output_tmp.fillna(0)
        scores_output_df = scores_output_df.add(score_output_tmp, fill_value=0)

        with open(f'{OUTPUT_DIR}/{i+1:03}_{book}/{CALC_NAME}_{DATASET_NAME}.csv', 'w') as f:
            scores_output_df.to_csv(f)

        pbar.update(1)


def is_target_word(feature):
    # if feature[0] == 'フィラー':
    #     return True
    # if feature[0] == '感動詞':
    #     return True
    # return False
    if is_fpp_word(feature):
        return False
    if feature[0] == 'BOS/EOS':
        return False
    if feature[0] == '記号':
        return False
    return True


def is_fpp_word(feature):
    if feature[0] == '名詞':
        if feature[1] == '代名詞':
            word = feature[7]
            if word in fpp_katakana:
                return True
        elif feature[1] == '一般':
            if len(feature) <= 7:
                pass
            elif feature[6] in fpp_norm:
                return True
            elif feature[7] in fpp_katakana:
                return True
    return False


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
