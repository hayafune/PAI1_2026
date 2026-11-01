#!/usr/bin/env python3
"""
panel_render.py -- パネルの画像描画と座標変換（ROS非依存）

次の3か所から共通で使用する。
  - panel_node.py（画像の配信）
  - train_cnn.py（学習データの生成）
  - perception_node.py（認識）

座標の約束:
  - ピクセル座標(u, v): パネル上の位置。単位はピクセル
    - 原点: パネル板全体の左上、+u: 右、+v: 下
  - パネル座標(y, z): パネル上の位置。単位はm
    - 原点: パネル板全体の中心、+y: 左、+z: 上
  - ロボット座標(x, y, z): RVizに表示される3次元空間上の位置。単位はm
    - 原点: ロボットのベースリンク(panda_link0)の位置、+x: ロボットの前方、+y: 左、+z: 上
  - パネル座標およびロボット座標の+y、+zの向きは一致している。
  - 画像はパネル板（9個のマス + 周囲の余白: PLATE_MARGIN）全体を写す
  - マス番号0..8は行優先（左上=0, 右上=2, 左下=6, 右下=8）。表示する数字は1..9
"""

import cv2
import numpy as np

PLATE_MARGIN = 0.02  # 格子の外側の余白(m)（RVizのパネル板と同じ値）


def plate_size(panel_size):
    """パネル板全体（外側の余白を含む）の一辺の長さ(m)を返す。

       panel_size: パネル板（外側の余白を含まない）の一辺の長さ(m)
    """
    return panel_size + 2.0 * PLATE_MARGIN


def pixels_per_meter(img_px, panel_size):
    """1 mあたりのピクセルの個数を返す。

       img_px:     パネル板全体（外側の余白を含む）の一辺のピクセルの個数
       panel_size: パネル板（外側の余白を含まない）の一辺の長さ(m)
    """
    return img_px / plate_size(panel_size)


def cell_center_px(row, col, img_px, panel_size):
    """マス(row, col)の中心のピクセル座標(u, v)を返す。

       img_px:     パネル板全体（外側の余白を含む）の一辺のピクセルの個数
       panel_size: パネル板（外側の余白を含まない）の一辺の長さ(m)
    """
    ppm = pixels_per_meter(img_px, panel_size)
    pitch_px = panel_size / 3.0 * ppm # マス1辺分のピクセルの個数
    u = img_px / 2.0 + (col - 1) * pitch_px
    v = img_px / 2.0 + (row - 1) * pitch_px
    return u, v


def pixel_to_panel(u, v, img_px, panel_size):
    """ピクセル座標(u, v) → パネル座標(dy, dz)。
       ピクセル座標は、原点がパネルの左上、左が-u、上が-v。
       パネル座標は、原点がパネルの中心、左が+y、上が+z。

       img_px:     パネル板全体（外側の余白を含む）の一辺のピクセルの個数。
       panel_size: パネル板（外側の余白を含まない）の一辺の長さ(m)。
    """
    ppm = pixels_per_meter(img_px, panel_size)
    dy = -(u - img_px / 2.0) / ppm
    dz = -(v - img_px / 2.0) / ppm
    return dy, dz


def draw_digit(img, digit, u, v, height_px,
               font=cv2.FONT_HERSHEY_SIMPLEX, color=(20, 20, 20), thick_scale=1.0):
    """数字digitを、中心(u, v)、高さheight_px（ピクセル）で画像img上に描く。

       color: 文字の色(BGR)。既定値はほぼ黒。
       thick_scale : 線の太さの倍率。学習データで太さを変動させてOpenCVの版による描画の差に強くするために使う。
    """
    text = str(digit)
    # https://imagingsolution.net/program/python/opencv-python/opencv-reference/opencv-python-gettextsize/
    (_, th), _ = cv2.getTextSize(text, font, 1.0, 1)
    scale = height_px / float(th)
    # 文字の太さ: 基準値を高さの9%とし、引数thick_scaleで調整する。
    thick = max(1, int(round(height_px * 0.09 * thick_scale)))
    # 描画する文字の幅および高さ(tw, th)（ピクセル）を得る
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    # OpenCVにおける文字描画の基準のピクセル座標（左下）を得る。右が+、下が+
    org = (int(round(u - tw / 2.0)), int(round(v + th / 2.0)))
    cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)


def render_panel(layout, img_px=360, panel_size=0.30,
                 font=cv2.FONT_HERSHEY_SIMPLEX,
                 jitter_px=0.0, noise_std=0.0, blur_ksize=0, rng=None,
                 height_scale=1.0, thick_scale=1.0):
    """パネル画像を描画して返す。data-augmentation付き。
    layout: 長さ9のリスト。layout[i] はマス番号i（行優先）の数字(1..9)。
    戻り値: BGRのuint8画像 (img_px x img_px x 3)

    img_px:     パネル板全体（外側の余白を含む）の一辺のピクセルの個数
    panel_size: パネル板（外側の余白を含まない）の一辺の長さ(m)

    jitter_px : 各数字の位置を±この画素数だけランダムにずらす（認識を難しくするため）
    noise_std : ガウスノイズの標準偏差（0..255のスケール）
    blur_ksize: ぼかしのカーネルサイズ（奇数。0はぼかしなし）
    rng: 乱数生成器
    height_scale: 数字の大きさの倍率
    thick_scale : 線の太さの倍率。学習データで太さを変動させてOpenCVの版による描画の差に強くするために使う。
    """
    if rng is None:
        rng = np.random.default_rng()  # 乱数生成器の設定
    img = np.full((img_px, img_px, 3), 235, np.uint8)  # 明るい灰色の板
    ppm = pixels_per_meter(img_px, panel_size) # 1 mあたりのピクセルの個数
    pitch_px = panel_size / 3.0 * ppm # マス1辺のピクセルの個数
    half = img_px / 2.0 # パネル板全体の中心のピクセル座標

    # 格子線を描く
    for k in range(4):
        # パネル板の中央から、-1.5, -0.5, 0.5, 1.5マスの位置に格子線を描画する。
        p = int(round(half + (k - 1.5) * pitch_px))
        # 一番端の線の位置
        lo = int(round(half - 1.5 * pitch_px))
        hi = int(round(half + 1.5 * pitch_px))
        cv2.line(img, (p, lo), (p, hi), (170, 170, 170), 1, cv2.LINE_AA) # 縦線
        cv2.line(img, (lo, p), (hi, p), (170, 170, 170), 1, cv2.LINE_AA) # 横線

    # 数字を描く
    for idx, digit in enumerate(layout):
        row, col = divmod(idx, 3) # linear to 2D (row-major)
        # マス(row, col)の中心のピクセル座標(u, v)
        u, v = cell_center_px(row, col, img_px, panel_size)
        # 数字の中心の位置をずらす
        if jitter_px > 0:
            u += rng.uniform(-jitter_px, jitter_px)
            v += rng.uniform(-jitter_px, jitter_px)
        # 文字の大きさの基準値をマス1辺の長さの55%とし引数height_scaleで調整する。
        draw_digit(img, digit, u, v, pitch_px * 0.55 * height_scale, font=font,
                   thick_scale=thick_scale)
    # ぼかしをかける
    if blur_ksize and blur_ksize > 1:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    # ノイズを加える
    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, img.shape)
        # np.clipで、[0,255]の範囲に調整する。
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def extract_cells(img_bgr, panel_size, out_px=32):
    """
    パネル画像img_bgrから9マスを切り出す。マス番号は行優先。
    戻り値: 9個のマスの画像(9, out_px, out_px)。型はfloat32、値は(0..1)。
           数字部分が大きい値になるよう加工済み。

    panel_size: パネル板（外側の余白を含まない）の一辺の長さ(m)
    out_px: 切り出す各マスの画像1辺のピクセルの個数
    学習(train_cnn.py)および認識(perception_node.py)で同じ関数を使う。
    """
    img_px = img_bgr.shape[0] # パネル板全体1辺のピクセルの個数
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) # BGR to gray-scale
    ppm = pixels_per_meter(img_px, panel_size) # 1 mあたりのピクセルの個数
    # 1マスの0.5辺のピクセルの個数。格子線を避けて少し内側を切り出すために0.92を掛ける
    half = panel_size / 3.0 * ppm / 2.0 * 0.92
    out = np.empty((9, out_px, out_px), np.float32)
    for idx in range(9):
        row, col = divmod(idx, 3) # linear to 2D (row-major)
        # マス(row, col)の中心のピクセル座標(u, v)
        u, v = cell_center_px(row, col, img_px, panel_size)
        x0, x1 = int(round(u - half)), int(round(u + half))
        y0, y1 = int(round(v - half)), int(round(v + half))
        crop = gray[y0:y1, x0:x1]
        # サイズ(out_px, out_px)に縮小または拡大
        crop = cv2.resize(crop, (out_px, out_px), interpolation=cv2.INTER_AREA)
        # 値を[0,1]に正規化するとともに、白黒を反転させて数字のピクセル値を大きくする。
        out[idx] = 1.0 - crop.astype(np.float32) / 255.0
    return out


if __name__ == "__main__":
    # 単体テスト: python3 panel_render.pyでpanel_test.pngを出力する。
    out = render_panel([1, 2, 3, 4, 5, 6, 7, 8, 9])
    cv2.imwrite("panel_test.png", out)
    print("wrote panel_test.png", out.shape)
