#!/usr/bin/env python3
"""
train_cnn.py -- Step 3: 仮想パネルの数字(1〜9)を認識する小さな CNN を学習する

TensorFlow (CPU) を入れた「別の仮想環境 (~/venv_tf)」で実行する。ROS の Python 環境には入れない。
学習結果は digit_cnn.npz に書き出し、ROS 側は NumPy だけ (digit_cnn_np.py) で読み込む。

学習データ:
  panel_render.render_panel() で、ランダムな配置・フォント・位置ずれ・ノイズ・ぼかし・明るさの
  パネル画像を作り、9マスを切り出して (画像, 数字) の組にする。実機カメラのデータは使わない。

使い方:
  source ~/venv_tf/bin/activate
  python train_cnn.py                       # 標準 (CPUで数分)
  python train_cnn.py --panels 600 --epochs 3   # 動作確認用の短縮版
"""

import argparse
import time

import cv2
import numpy as np

import panel_render as pr

FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,   # panel_node.py の既定フォント
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
]


def make_dataset(n_panels, rng, panel_size=0.30, img_px=360, cell_px=32, augment=True):
    xs, ys = [], []
    for _ in range(n_panels):
        layout = [int(d) for d in rng.permutation(np.arange(1, 10))]
        if augment:
            font = FONTS[int(rng.integers(len(FONTS)))]
            img = pr.render_panel(
                layout, img_px, panel_size, font=font,
                jitter_px=float(rng.uniform(0, 12)),
                noise_std=float(rng.uniform(0, 12)),
                blur_ksize=int(rng.choice([0, 3, 5])),
                rng=rng,
                height_scale=float(rng.uniform(0.9, 1.1)),   # 数字の大きさの変動
                thick_scale=float(rng.uniform(0.7, 1.5)),    # 線の太さの変動 (OpenCV の版差の吸収)
            )
            gain = rng.uniform(0.8, 1.1)       # 明るさ・コントラストの変動
            bias = rng.uniform(-25, 15)
            img = np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        else:
            img = pr.render_panel(layout, img_px, panel_size, rng=rng)
        cells = pr.extract_cells(img, panel_size, cell_px)      # (9, 32, 32)
        xs.append(cells)
        ys.extend(d - 1 for d in layout)                        # クラス0..8 = 数字1..9
    x = np.concatenate(xs, axis=0)[..., None].astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    return x, y


def build_model(keras):
    return keras.Sequential([
        keras.layers.Input((32, 32, 1)),
        keras.layers.Conv2D(8, 3, activation="relu"),    # 32 -> 30
        keras.layers.MaxPooling2D(2),                    # -> 15
        keras.layers.Conv2D(16, 3, activation="relu"),   # -> 13
        keras.layers.MaxPooling2D(2),                    # -> 6
        keras.layers.Flatten(),                          # 6*6*16 = 576
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(9),                           # ロジット (softmax は損失側で)
    ])


def export_npz(model, path):
    conv1, _, conv2, _, _, fc1, fc2 = model.layers
    np.savez(
        path,
        conv1_w=conv1.get_weights()[0], conv1_b=conv1.get_weights()[1],
        conv2_w=conv2.get_weights()[0], conv2_b=conv2.get_weights()[1],
        fc1_w=fc1.get_weights()[0], fc1_b=fc1.get_weights()[1],
        fc2_w=fc2.get_weights()[0], fc2_b=fc2.get_weights()[1],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", type=int, default=2500, help="学習用パネル枚数 （1枚=9マス）")
    ap.add_argument("--test-panels", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="digit_cnn.npz")
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    x_train, y_train = make_dataset(args.panels, rng)
    x_test, y_test = make_dataset(args.test_panels, np.random.default_rng(args.seed + 1))
    print(f"data: train {x_train.shape}, test {x_test.shape}  ({time.time() - t0:.1f}s)")

    model = build_model(keras)
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, validation_split=0.1,
              epochs=args.epochs, batch_size=args.batch, verbose=2)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"test accuracy （別のseedで生成したデータ）: {acc * 100:.2f}%")

    export_npz(model, args.out)
    print(f"saved {args.out}")

    # NumPy 版の推論が Keras と一致するか確認 (ROS 側はこちらを使う)
    from digit_cnn_np import DigitCNN
    net = DigitCNN(args.out)
    ref = tf.nn.softmax(model.predict(x_test[:500], verbose=0)).numpy()
    mine = net.predict_proba(x_test[:500])
    print(f"NumPy版とKeras版の確率の最大差: {np.abs(ref - mine).max():.2e}")
    print(f"NumPy版の正解率: {(mine.argmax(1) == y_test[:500]).mean() * 100:.2f}% （500マス）")


if __name__ == "__main__":
    main()
