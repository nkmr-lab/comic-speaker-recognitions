
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import settings

DATASET_DIR     = 'data/dataset'
OUTPUT_DIR      = 'data/scores'
CALC_NAME       = Path(__file__).stem.lstrip('calc_')
DATASET_NAME    = settings.DATASET_NAME
manga109_parser = settings.manga109_parser


def main():

    pbar = tqdm(total=len(manga109_parser.books))
    for i, book in enumerate(manga109_parser.books):
        annotation, dataset, characters_target = init(i, book)

        # 出力用のDFを作成
        scores_df = pd.DataFrame(columns=characters_target)

        for annotation_page in annotation['page']:
            texts = annotation_page['text']
            bodys = annotation_page['body']
            faces = annotation_page['face']
            frames = annotation_page['frame']

            for text in texts:
                if text['@id'] not in dataset['annotation_id'].values:
                    continue

                inframes = filter(lambda frame: is_inframe(frame, text), frames)
                characters_inframe = set()
                for frame in inframes:
                    bodys_inframe = filter(lambda body: is_inframe(frame, body), bodys)
                    faces_inframe = filter(lambda face: is_inframe(frame, face), faces)
                    bodys_chara = {body['@character'] for body in bodys_inframe}
                    faces_chara = {face['@character'] for face in faces_inframe}
                    characters_inframe = characters_inframe | bodys_chara | faces_chara
                characters_inframe = characters_inframe & set(characters_target)  # ターゲットに無いキャラクタを省く
                score = 1.0 / len(characters_inframe) if len(characters_inframe) != 0 else 0
                scores_se = pd.Series(score, index=characters_inframe, name=text['@id'])
                scores_df = scores_df.append(scores_se)

            pbar.set_postfix(title=book, page=f'{annotation_page["@index"] + 1}/{len(annotation["page"])}')

        scores_df = scores_df.fillna(0.0)
        with open(f'{OUTPUT_DIR}/{i+1:03}_{book}/{CALC_NAME}_{DATASET_NAME}.csv', 'w') as f:
            scores_df.to_csv(f)

        pbar.update(1)


# frameの矩形とROI（body・face・text）の矩形の当たり判定
def is_inframe(frame, roi):
    x, y = calc_box_center(roi)
    if x >= frame['@xmin'] and x <= frame['@xmax']:
        if y >= frame['@ymin'] and y <= frame['@ymax']:
            return True
    return False


# 矩形の中心を求める
def calc_box_center(roi):
    x_center = (roi['@xmin'] + roi['@xmax']) / 2
    y_center = (roi['@ymin'] + roi['@ymax']) / 2
    return x_center, y_center


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
