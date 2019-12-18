# セリフ周辺のキャラクタから計算

import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import manga109api
from dotenv import load_dotenv
load_dotenv()

DATASET_DIR = 'data/dataset'
OUTPUT_DIR = 'data/scores'
manga109_parser = manga109api.Parser(root_dir=os.environ['MANGA109_ROOT_DIR'])


def main():
    for i, book in enumerate(tqdm(manga109_parser.books)):
        # データセット読み込み
        annotation = manga109_parser.annotations[book]
        dataset_path = Path(f'{DATASET_DIR}/{i+1:03}_{book}.csv')
        if dataset_path.exists() is False:
            continue
        with open(dataset_path, 'r') as f:
            dataset = pd.read_csv(f)
        with open(f'{DATASET_DIR}/{i+1:03}_{book}_character.txt', 'r') as f:
            characters_target = f.read()
        characters_target = characters_target.split('\n')

        # 出力用のDFを作成
        scores_df = pd.DataFrame(columns=characters_target)

        # HACK: データセットのいらない情報まで入っててキモい
        for annotation_page in annotation['book']['pages']['page']:
            page = annotation_page['@index']
            if 'text' not in annotation_page:
                continue
            texts = annotation_page['text']

            bodys = annotation_page['body'] if 'body' in annotation_page else []
            faces = annotation_page['face'] if 'face' in annotation_page else []

            # 外側が配列じゃないことがあることの対策
            texts = texts if type(texts) is list else [texts]
            bodys = bodys if type(bodys) is list else [bodys]
            faces = faces if type(faces) is list else [faces]

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
                if annotation_id not in dataset[dataset.page == page]['annotation_id'].values:
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

                per = per_both['face'] * 1.5 + per_both['body']
                if per.sum() != 0:
                    per_norm = per / per.sum()
                else:
                    per_norm = pd.Series(1.0 / len(characters_target), index=characters_target)
                per_norm = per_norm.rename(annotation_id)
                scores_df = scores_df.append(per_norm)

        with open(f'{OUTPUT_DIR}/{i+1:03}_{book}_neighbor.csv', 'w') as f:
            scores_df.to_csv(f)


# バウンディングボックスの中心を求める
def calc_box_center(x_min, y_min, x_max, y_max):
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return x_center, y_center


if __name__ == '__main__':
    main()
