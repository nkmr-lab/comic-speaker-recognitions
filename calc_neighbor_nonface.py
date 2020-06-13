# セリフ周辺のキャラクタから計算

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import settings

DATASET_DIR     = 'data/dataset'
OUTPUT_DIR      = 'data/scores'
CALC_NAME       = Path(__file__).stem.lstrip('calc_')
DATASET_NAME    = settings.DATASET_NAME
manga109_parser = settings.manga109_parser


def main():
    books = manga109_parser.books
    pbar = tqdm(total=len(books))
    for i, book in enumerate(books):
        # データセット読み込み
        annotation = manga109_parser.get_annotation(book)
        dataset_path = Path(f'{DATASET_DIR}/{i+1:03}_{book}.csv')
        with open(dataset_path, 'r') as f:
            dataset = pd.read_csv(f)
        with open(f'{DATASET_DIR}/{i+1:03}_{book}_character.txt', 'r') as f:
            characters_target = f.read()
        characters_target = characters_target.split('\n')

        # 出力用のDFを作成
        scores_df = pd.DataFrame(columns=characters_target)

        # HACK: データセットのいらない情報まで入っててキモい
        for annotation_page in annotation['page']:
            texts = annotation_page['text']

            bodys = annotation_page['body']
            faces = annotation_page['face']

            # ボディのポジションとキャラ名
            bodys_pos = np.array([calc_box_center(b['@xmin'], b['@ymin'], b['@xmax'], b['@xmax']) for b in bodys])
            bodys_pos = bodys_pos if len(bodys_pos) != 0 else np.empty((0, 2))  # 距離計算用に配列が空だったときの形を変える
            bodys_chara = [b['@character'] for b in bodys]
            # フェイスのポジションとキャラ名
            faces_pos = np.array([calc_box_center(f['@xmin'], f['@ymin'], f['@xmax'], f['@xmax']) for f in faces])
            faces_pos = faces_pos if len(faces_pos) != 0 else np.empty((0, 2))
            faces_chara = [f['@character'] for f in faces]

            # 距離の最大値計算
            width, height = annotation_page['@width'], annotation_page['@height']
            dist_max = np.linalg.norm([width, height])

            for text in texts:
                annotation_id = text['@id']
                if annotation_id not in dataset['annotation_id'].values:
                    continue

                # テキストのポジション
                text_pos = np.array(calc_box_center(text['@xmin'], text['@ymin'], text['@xmax'], text['@ymax']))
                bodys_dist = np.linalg.norm(bodys_pos - text_pos, axis=1)
                faces_dist = np.linalg.norm(faces_pos - text_pos, axis=1)

                body_dist = pd.Series(bodys_dist, index=bodys_chara).groupby(level=0).min()
                face_dist = pd.Series(faces_dist, index=faces_chara).groupby(level=0).min()

                per_body = 1 - (body_dist / dist_max)
                per_face = 1 - (face_dist / dist_max)
                per_both = pd.DataFrame({'body': per_body, 'face': per_face}, index=characters_target)  # index指定によりキャラを制限
                per_both = per_both.fillna(0)

                # per = per_both['face'] * 1.5 + per_both['body']

                # 得点が最大の方にすることにした
                # per_both['face'] = per_both['face'] * 1.5
                per = per_both.max(axis=1)

                if per.sum() != 0:
                    per_norm = per / per.sum()
                else:
                    # per_norm = pd.Series(1.0 / len(characters_target), index=characters_target)
                    per_norm = pd.Series(0, index=characters_target)
                per_norm = per_norm.rename(annotation_id)
                scores_df = scores_df.append(per_norm)

            pbar.set_postfix(title=book, page=f'{annotation_page["@index"]+1}/{len(annotation["page"])}')

        with open(f'{OUTPUT_DIR}/{i+1:03}_{book}/{CALC_NAME}_{DATASET_NAME}.csv', 'w') as f:
            scores_df.to_csv(f)

        pbar.update(1)


# バウンディングボックスの中心を求める
def calc_box_center(x_min, y_min, x_max, y_max):
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return x_center, y_center


if __name__ == '__main__':
    main()
