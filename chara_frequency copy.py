
import pandas as pd
from tqdm import tqdm
import settings


manga109_parser = settings.manga109_parser
TOP_N = 5
OTHERS = ['_Other', 'Ｏｈｔｅｒ', 'other', 'Other', 'Ｏｔｈｅｒ', 'Others', 'the other']


def main():
    books = manga109_parser.books
    output_df = pd.DataFrame([], columns=['book_id', 'character_id'])
    pbar = tqdm(total=len(books))
    for i, book in enumerate(books):
        annotation = manga109_parser.get_annotation(book)

        # otherのキャラidを確認
        others = list(filter(lambda x: x['@name'] in OTHERS, annotation['character']))
        others_id = [o['@id'] for o in others]
        # print(others_id)

        # faceおよびbodyに登場するキャラクタの配列
        faces_chara = [roi['@character'] for p in annotation['page'] for roi in p['face'] if roi['@character'] not in others_id]
        bodys_chara = [roi['@character'] for p in annotation['page'] for roi in p['body'] if roi['@character'] not in others_id]
        charas = faces_chara + bodys_chara
        frequency = pd.Series(charas).value_counts()
        # print(frequency)

        top_character = frequency.index[:TOP_N].values
        output_df = pd.concat([output_df, pd.DataFrame({'book_id': i + 1, 'character_id': top_character})])

        pbar.update(1)

    with open('data/chara_threthold_freq.csv', 'w') as f:
        output_df.to_csv(f, index=None)

        # texts = [{'text': t['#text'], 'id': t['@id']} for p in annotation['page'] for t in p['text']]


if __name__ == '__main__':
    main()
