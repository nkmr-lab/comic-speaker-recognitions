
import numpy as np
from scipy import stats
from scipy import signal
from scipy.interpolate import UnivariateSpline
from PIL import Image
import cv2
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import settings

DATASET_DIR     = 'data/dataset'
OUTPUT_DIR      = 'data/scores'
CALC_NAME       = Path(__file__).stem.lstrip('calc_')
DATASET_NAME    = settings.DATASET_NAME
manga109_parser = settings.manga109_parser
LENGTH_T = 0.1  # 伸ばす距離の比率


def main():
    books = manga109_parser.books
    pbar = tqdm(total=len(books))
    for i, book in enumerate(books):
        annotation, dataset, characters_target = init(i, book)

        scores_output_df = pd.DataFrame(0, index=dataset['annotation_id'].values, columns=characters_target)
        scores_df = pd.DataFrame(columns=characters_target, )

        for page in annotation['page']:
            texts = page['text']
            bodys = page['body']
            faces = page['face']
            img = Image.open(manga109_parser.img_path(book=book, index=page['@index']))
            img = np.array(img)

            for text in texts:
                if text['@id'] not in dataset['annotation_id'].values:
                    continue
                # print(text)
                # ここから計算スタート
                text_bbox = (text['@xmin'], text['@xmax'], text['@ymin'], text['@ymax'])
                try:
                    bubble_contour = get_bubble_contour(img, text_bbox)
                except:
                    print('err')
                    continue
                # 適当だが，ありえないほど輪郭が長いやつはバルーンじゃないとして弾く（これが速度を遅くする原因）
                if len(bubble_contour) > 2000:
                    continue
                # 少なすぎてもだめ
                if len(bubble_contour) < 10:
                    continue
                x = bubble_contour[:, 0]
                y = bubble_contour[:, 1]
                curv = curvature_splines(x, y)
                maxid = signal.argrelmax(curv, order=10)
                minid = signal.argrelmin(curv, order=10)
                try:
                    # もししっぽと判断されなければ飛ばす
                    if not is_tail(curv[minid]):
                        continue
                except:
                    print('err')
                    continue

                # 候補の3頂点を探す
                id_vertex = curv.argmin()
                id_subver1 = [i for i in maxid[0] if i < id_vertex]
                id_subver2 = [i for i in maxid[0] if i > id_vertex]
                if len(id_subver1) == 0 or len(id_subver2) == 0:
                    continue
                id_subver1 = id_subver1[-1]
                id_subver2 = id_subver2[0]

                # 頂点と中点
                vertex = np.array([x[id_vertex], y[id_vertex]])
                center = np.array([(x[id_subver1] + x[id_subver2]) / 2, (y[id_subver1] + y[id_subver2]) / 2])

                w = np.linalg.norm([annotation['page'][0]['@width'], annotation['page'][0]['@height']]) * LENGTH_T
                endpoint = calc_direction(vertex, center, w)

                # 顔および体から当たり判定を全探索
                speaker_charas = set()
                for rois in [bodys, faces]:
                    for roi in rois:
                        if is_in_bounding(vertex, endpoint, roi):
                            speaker_charas.add(roi['@character'])

                speaker_charas = speaker_charas & set(characters_target)
                score = 1.0 / len(speaker_charas) if len(speaker_charas) != 0 else 0
                scores_se = pd.Series(score, index=speaker_charas, name=text['@id'])
                scores_df = scores_df.append(scores_se)
                # print(speaker_charas)

            pbar.set_postfix(page=f'{page["@index"] + 1}/{len(annotation["page"])}')

        scores_df = scores_df.fillna(0.0)
        scores_output_df = scores_output_df.add(scores_df, fill_value=0)
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


# 入力:画像とテキストの位置(xmin,xmax,ymin)の順のタプル
# 出力:輪郭の座標（x,y）の配列
# FIXME: 吹き出しラベルの1つ外側の黒線も取得しないと，吹き出し方向が取れないものがある
def get_bubble_contour(img, text_bbox):

    xmin, xmax, ymin, ymax = text_bbox

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津の二値化

    # ラベリングの処理
    _, label = cv2.connectedComponents(gray)

    # テキストが存在する中で，0（背景ラベル）を除く最も数が多いラベルを取得
    # FIXME: どうやらbubble_labelがからになることがあるらしい
    trim_label = label[ymin:ymax, xmin:xmax]
    bubble_label, _ = stats.mode(trim_label[trim_label.nonzero()], axis=None)
    bubble_label = bubble_label[0]

    # 対象の吹き出しのみの2値化画像を作る
    bubble_gray_img = np.zeros(label.shape, dtype='uint8')
    bubble_gray_img[label == bubble_label] = 255

    # 最も外側の輪郭のみ取得する
    contours, _ = cv2.findContours(bubble_gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bubble_contours = contours[0].squeeze()

    return bubble_contours


# 曲率計算
def curvature_splines(x, y=None, error=0.1):
    """Calculate the signed curvature of a 2D curve at each point
    using interpolating splines.
    Parameters
    ----------
    x,y: numpy.array(dtype=float) shape (n_points, )
         or
         y=None and
         x is a numpy.array(dtype=complex) shape (n_points, )
         In the second case the curve is represented as a np.array
         of complex numbers.
    error : float
        The admisible error when interpolating the splines
    Returns
    -------
    curvature: numpy.array shape (n_points, )
    Note: This is 2-3x slower (1.8 ms for 2000 points) than `curvature_gradient`
    but more accurate, especially at the borders.
    """

    # handle list of complex case
    if y is None:
        x, y = x.real, x.imag

    t = np.arange(x.shape[0])
    std = error * np.ones_like(x)

    fx = UnivariateSpline(t, x, k=4, w=1 / np.sqrt(std))
    fy = UnivariateSpline(t, y, k=4, w=1 / np.sqrt(std))

    xˈ = fx.derivative(1)(t)
    xˈˈ = fx.derivative(2)(t)
    yˈ = fy.derivative(1)(t)
    yˈˈ = fy.derivative(2)(t)
    curvature = (xˈ * yˈˈ - yˈ * xˈˈ) / np.power(xˈ ** 2 + yˈ ** 2, 3 / 2)
    return curvature


# 局所最小値を入れてしっぽか判断
def is_tail(curv_mins):
    curv_minimam = curv_mins.min()  # 頂点候補
    curv_mins_other = curv_mins[np.where(curv_mins != curv_minimam)]
    curv_mins_mean = curv_mins_other.mean()
    per = np.abs(curv_mins_mean) / np.abs(curv_minimam)
    if per < 0.080:
        return True
    return False


# 線分の端を求める
def calc_direction(vertex, center, w):
    vector = (vertex - center) / np.linalg.norm(vertex - center)
    move = w * vector
    endpoint = vertex + move
    return endpoint


# 線分交差判定
def intersect(p1, p2, p3, p4):
    tc1 = (p1[0] - p2[0]) * (p3[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p3[0])
    tc2 = (p1[0] - p2[0]) * (p4[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p4[0])
    td1 = (p3[0] - p4[0]) * (p1[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p1[0])
    td2 = (p3[0] - p4[0]) * (p2[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p2[0])
    return tc1 * tc2 < 0 and td1 * td2 < 0


# 矩形と線分の交差判定
def is_in_bounding(vertex, endpoint, b):
    lt = [b['@xmin'], b['@ymin']]
    rt = [b['@xmax'], b['@ymin']]
    rb = [b['@xmax'], b['@ymax']]
    lb = [b['@xmin'], b['@ymax']]

    if intersect(vertex, endpoint, lt, rt):
        return True
    if intersect(vertex, endpoint, rt, rb):
        return True
    if intersect(vertex, endpoint, rb, lb):
        return True
    if intersect(vertex, endpoint, lb, lt):
        return True

    return False


if __name__ == '__main__':
    main()
