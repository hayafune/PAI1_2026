#!/usr/bin/env python3
"""
eval_offline.py -- ROS を使わずに、画像の条件別の認識精度を調べる診断スクリプト

「ROS上の正解率が低い原因が、(a) 画像が学習時より劣化しているのか、(b) ROS側の問題なのか」を切り分ける。
panel_render.py / digit_cnn_np.py / digit_cnn.npz だけを使う （TensorFlowもROSも不要）。

使い方:
  python3 eval_offline.py                 # 各条件100枚
  python3 eval_offline.py --panels 300
"""

import argparse
import itertools

import numpy as np

import panel_render as pr
from digit_cnn_np import DigitCNN

# 学習時の範囲: jitter 0〜12 px, noise 0〜12, blur {0,3,5}, 明るさ0.8〜1.1倍 / -25〜+15,
#               太さ0.7〜1.5倍, 大きさ0.9〜1.1倍
SCENARIOS = [
    ("clean (panel_node の既定)",          dict(jitter_px=0, noise_std=0, blur_ksize=0)),
    ("学習範囲の上限",                       dict(jitter_px=8, noise_std=12, blur_ksize=5)),
    ("noise_std=20 のみ",                   dict(jitter_px=0, noise_std=20, blur_ksize=0)),
    ("blur_ksize=7 のみ",                   dict(jitter_px=0, noise_std=0, blur_ksize=7)),
    ("jitter_px=10 のみ",                   dict(jitter_px=10, noise_std=0, blur_ksize=0)),
    ("劣化した画像 (20 / 7 / 10)",       dict(jitter_px=10, noise_std=20, blur_ksize=7)),
    ("線を太く×1.5・大きく×1.1",             dict(thick_scale=1.5, height_scale=1.1)),
    ("線を細く×0.7・小さく×0.9",             dict(thick_scale=0.7, height_scale=0.9)),
    # ---- 以下は学習範囲を超える「限界テスト」。どこで raw が崩れ、decoded が救えるかを探す ----
    ("限界: noise_std=40",                  dict(noise_std=40)),
    ("限界: blur_ksize=11",                 dict(blur_ksize=11)),
    ("限界: jitter_px=16",                  dict(jitter_px=16)),
    ("限界: 40 / 9 / 14",                   dict(jitter_px=14, noise_std=40, blur_ksize=9)),
    ("限界: 60 / 11 / 16",                  dict(jitter_px=16, noise_std=60, blur_ksize=11)),
]


# perception_node.py と同じ復号 （1〜9が1回ずつという制約条件）
def build_permutations():
    return np.array(list(itertools.permutations(range(9))), dtype=np.int8)


def decode_unique(probs, perms):
    logp = np.log(np.clip(probs, 1e-9, 1.0))
    scores = logp[np.arange(9)[None, :], perms].sum(axis=1)
    return perms[int(np.argmax(scores))].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", type=int, default=100)
    ap.add_argument("--model", default="digit_cnn.npz")
    ap.add_argument("--panel-size", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    net = DigitCNN(args.model)
    perms = build_permutations()
    rng = np.random.default_rng(args.seed)

    print(f"{'条件':<32} {'マス正解率 raw':>14} {'decoded':>9} {'パネル正解率 raw':>16} {'decoded':>9}")
    for name, kw in SCENARIOS:
        cell_raw = cell_dec = panel_raw = panel_dec = 0
        for _ in range(args.panels):
            layout = [int(d) for d in rng.permutation(np.arange(1, 10))]
            img = pr.render_panel(layout, 360, args.panel_size, rng=rng, **kw)
            probs = net.predict_proba(pr.extract_cells(img, args.panel_size))
            truth = np.array(layout)
            raw = probs.argmax(1) + 1
            dec = decode_unique(probs, perms) + 1
            cell_raw += int((raw == truth).sum())
            cell_dec += int((dec == truth).sum())
            panel_raw += int((raw == truth).all())
            panel_dec += int((dec == truth).all())
        n = args.panels
        print(f"{name:<32} {100 * cell_raw / (9 * n):>13.1f}% {100 * cell_dec / (9 * n):>8.1f}% "
              f"{100 * panel_raw / n:>15.1f}% {100 * panel_dec / n:>8.1f}%")


if __name__ == "__main__":
    main()
