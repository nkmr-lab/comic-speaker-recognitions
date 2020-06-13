
import numpy as np
from scipy import stats
from scipy import signal
from scipy.interpolate import UnivariateSpline
from PIL import Image
import cv2
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import math
import settings

DATASET_DIR     = 'data/dataset'
OUTPUT_DIR      = 'data/scores'
CALC_NAME       = Path(__file__).stem.lstrip('calc_')
DATASET_NAME    = settings.DATASET_NAME
manga109_parser = settings.manga109_parser
LENGTH_T = 0.05  # 伸ばす距離の比率
TOP_DEG = 30  # 広げる角度


def main():
    books = manga109_parser.books
    pbar = tqdm(total=len(books))
    for i, book in enumerate(books):
        annotation, dataset, characters_target = init(i, book)

        scores_output_df = pd.DataFrame(0, index=dataset['annotation_id'].values, columns=characters_target)
        scores_df = pd.DataFrame(columns=characters_target, )

        w = np.linalg.norm([annotation['page'][0]['@width'], annotation['page'][0]['@height']]) * LENGTH_T

        for page in annotation['page']:
            texts = page['text']
            bodys = page['body']
            faces = page['face']
            frames = page['frame']
            img = Image.open(manga109_parser.img_path(book=book, index=page['@index']))
            img = np.array(img)

            gray = conv_gray(img)
            thresh = conv_thresh(gray)
            labels = get_labels(thresh)

            # img_copy = img.copy()

            for text in texts:
                if text['@id'] not in dataset['annotation_id'].values:
                    continue

                # ここから計算スタート
                try:
                    contours = get_balloon_contours(labels, text)
                except:
                    print('err')
                    continue

                if is_balloon(contours, text) is False:
                    continue

                try:
                    curve = curvature_splines(contours[:, 0], contours[:, 1])
                    if curve.mean() > 0:
                        curve = curve * -1
                except:
                    continue

                target_frame = whitemap_target_frame(text, frames, thresh.shape, thickness=15)
                target_bool = target_frame[contours[:, 1], contours[:, 0]] == 0

                contours_target = contours[target_bool]
                curve_target = curve[target_bool]

                try:
                    tail_index = calc_tail_index(curve_target)
                except:
                    continue
                if tail_index is None:
                    continue

                left_index, right_index = calc_vertex_index(curve_target, tail_index)
                if left_index is None:
                    continue
                vertex_top = contours_target[tail_index]
                vertex_left = contours_target[left_index]
                vertex_right = contours_target[right_index]
                line_center = np.array([(vertex_left[0] + vertex_right[0]) / 2, (vertex_left[1] + vertex_right[1]) / 2])

                if is_tail2(vertex_top, vertex_left, vertex_right, curve_target) is False:
                    continue

                endpoint = calc_direction(vertex_top, line_center, w)

                """ ここから追加分 """
                line_size = np.linalg.norm(endpoint - vertex_top)
                vector = (endpoint - vertex_top) / line_size
                vertical = np.array([vector[1], vector[0] * -1])

                vertical_size = line_size * math.tan(math.radians(TOP_DEG / 2))

                vertex_2 = endpoint + (vertical * vertical_size)
                vertex_3 = endpoint + (vertical * vertical_size * -1)

                # cv2.ellipse(img_copy, tuple(vertex_top), (8, 8), 0, 0, 360, (255, 0, 0), thickness=2)
                # cv2.line(img_copy, tuple(vertex_top), tuple(endpoint.astype('int')), (0, 255, 0), thickness=2)

                # 顔および体から当たり判定を全探索
                speaker_charas = set()
                for rois in [bodys, faces]:
                    for roi in rois:
                        for points in [[vertex_top, vertex_2], [vertex_2, vertex_3], [vertex_3, vertex_top]]:
                            p1 = points[0]
                            p2 = points[1]
                            if is_in_bounding(p1, p2, roi):
                                speaker_charas.add(roi['@character'])

                speaker_charas = speaker_charas & set(characters_target)
                score = 1.0 / len(speaker_charas) if len(speaker_charas) != 0 else 0
                scores_se = pd.Series(score, index=speaker_charas, name=text['@id'])
                scores_df = scores_df.append(scores_se)
                # print(speaker_charas)

            pbar.set_postfix(page=f'{page["@index"] + 1}/{len(annotation["page"])}')
            # Image.fromarray(np.uint8(img_copy)).save(f'notebooks/tmp_img/is_tail/{book}_{page["@index"]:03}.jpg')

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


def conv_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def conv_thresh(gray):
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津の二値化
    return thresh

def get_labels(thresh):
    _, labels = cv2.connectedComponents(thresh)
    return labels

# テキストが所属するフレームを返す
def whitemap_target_frame(text, frames, img_shape, thickness=10):
    frame_img = np.zeros(img_shape, dtype='uint8')
    inframe = list(filter(lambda frame: is_inframe(frame, text), frames))
    if len(inframe) == 0:
        return frame_img
    for f in inframe:
        cv2.rectangle(frame_img, (f['@xmin'], f['@ymin']), (f['@xmax'], f['@ymax']), 255, thickness=thickness)
    return frame_img

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


def get_balloon_contours(labels, text):
    # 座標をもとに対象領域のラベル情報を取得
    xmin, xmax, ymin, ymax = text['@xmin'], text['@xmax'], text['@ymin'], text['@ymax']
    labels_target = labels[ymin:ymax, xmin:xmax]
    
    # ラベルの最頻値
    balloon_label, _ = stats.mode(labels_target[labels_target != 0])
    if len(balloon_label) == 0:
        return np.empty((0, 0))
    balloon_label = balloon_label[0]
    
    # 吹き出しのみの二値画像
    balloon_img = np.zeros(labels.shape, dtype='uint8')
    balloon_img[labels == balloon_label] = 255

    contours, _ = cv2.findContours(balloon_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0].squeeze()
    
    if contours.ndim != 2:
        return np.empty((0, 0))
    
    return contours


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


def is_balloon(contours, text):
    # そもそも空のとき
    if len(contours) == 0:
        return False
    
    # ベーシックな情報の抽出
    xmin, xmax, ymin, ymax = text['@xmin'], text['@xmax'], text['@ymin'], text['@ymax']
    bx, by = contours[:, 0], contours[:, 1]
    b_xmin, b_xmax, b_ymin, b_ymax = bx.min(), bx.max(), by.min(), by.max()
    
    text_w, text_h = xmax - xmin, ymax - ymin
    balloon_w, balloon_h = b_xmax - b_xmin, b_ymax - b_ymin
    
    # 輪郭線が長すぎたら吹き出しとみなさない
    if len(contours) > (text_w * 2 + text_h * 2) * 4:
        return False
    
    return True


# 曲率をもとに，tailになるindexを取得する
def calc_tail_index(curve):
    order = int(len(curve) / 10)
    minid = signal.argrelmin(curve, order=order)
    
    curve_mins = curve[minid]    
    
    if is_tail(curve_mins) is False:
        return None
    tail_index = minid[0][curve_mins.argmin()]
    return tail_index

# 局所最小値を入れてしっぽか判断
def is_tail(curve_mins):
    if len(curve_mins) == 0:
        return False
    curve_minimam = curve_mins.min()  # 頂点候補
    curve_mins_other = curve_mins[np.where(curve_mins != curve_minimam)]
    
#     if curve_minimam < 0.05:
#         return False
    
    # 対立候補がなく，一箇所だけ最小値を取ってればしっぽとする
    if len(curve_mins_other) == 0:
        return True

    curve_mins_mean = curve_mins_other.mean()
    per = np.abs(curve_mins_mean) / np.abs(curve_minimam)
    
    # 頂点の候補が多すぎず，他の頂点の平均値が最小値の半分未満だったとき
    if len(curve_mins_other) < 5 and per < 0.5:
        return True
    return False


def is_tail2(vertex_top, vertex_left, vertex_right, curve):
    
    u = vertex_left - vertex_top
    v = vertex_right - vertex_top
    x = np.inner(u, v)
    
    s = np.linalg.norm(u)
    t = np.linalg.norm(v)
    cos = x/(s*t)
    deg = np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0)))
    if deg > 125:
        return False
    if s > len(curve) * 0.2 or t > len(curve) * 0.2:
        return False
    return True


# 線分の端を求める
def calc_direction(vertex, center, w):
    vector = (vertex - center) / np.linalg.norm(vertex - center)
    move = w * vector
    endpoint = vertex + move
    return endpoint


# 根本の座標を求める
def calc_vertex_index(curve, tail_index):
    curve_diff = np.diff(curve)
    curve_diff_pos = np.where(curve_diff > 0)[0]
    curve_diff_neg = np.where(curve_diff < 0)[0]
    
    try:
        left_vertex_id = curve_diff_pos[curve_diff_pos < tail_index][-1] + 1
        right_vertex_id = curve_diff_neg[curve_diff_neg > tail_index][0]
    except:
        return None, None
    return left_vertex_id, right_vertex_id


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
