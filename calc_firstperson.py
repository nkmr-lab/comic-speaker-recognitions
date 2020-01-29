
import MeCab
import settings

CORPUS_PATH = 'data/corpus/fpp.csv'
manga109_parser = settings.manga109_parser
mecab = MeCab.Tagger(f'-d {settings.MECAB_IPADIC_PATH}')


def main():
    # コーパスを取得
    with open(CORPUS_PATH, 'r') as f:
        fpp = f.read().split()[1:]
    fpp_katakana = [x.split(',')[0] for x in fpp]
    fpp_norm = [x.split(',')[1] for x in fpp]

    books = manga109_parser.books
    for i, book in enumerate(books[:9]):
        annotation = manga109_parser.get_annotation(book)

        # 1つのコミックの全てのテキストとidを取得
        texts = [{'text': t['#text'], 'id': t['@id']} for p in annotation['page'] for t in p['text']]
        fpp_incomic = set()
        for text in texts:
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

        print(fpp_incomic)
        # print(list(filter(lambda x: x['fpp'] != '', texts)))
        print(len(list(filter(lambda x: x['fpp'] != '', texts))))
        print(len(texts))


if __name__ == '__main__':
    main()
