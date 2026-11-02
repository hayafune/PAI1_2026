#!/usr/bin/env python3
"""
train_cnn.py -- パネル画像の数字(1〜9)を認識する小さなCNNを学習する

TensorFlow(CPU)入りのPython仮想環境(~/venv_tf)で実行する。ROSのPython環境には入れない。
学習結果は重みファイルdigit_cnn.npzに書き出し、ROS側はNumPyだけのCNN(digit_cnn_np.py)で使用する。

学習データ:
  panel_render.render_panel()で、ランダムな配置・フォント・位置ずれ・ノイズ・ぼかし・明るさのパネル画像を作り、9マスを切り出して（画像, 数字）の組にする。実機カメラのデータは使わない。

使い方:
  source ~/venv_tf/bin/activate
  python train_cnn.py                           # 標準（CPUで数分）
  python train_cnn.py --panels 600 --epochs 3   # 動作確認用の短縮版
"""

import argparse
import time

import cv2
import numpy as np

import panel_render as pr

FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,   # panel_node.pyのフォントの既定値
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
                jitter_px=float(rng.uniform(0, 12)), # 位置ずれ
                noise_std=float(rng.uniform(0, 12)), # 各ピクセルの変動（ノイズ）
                blur_ksize=int(rng.choice([0, 3, 5])), # ぼかしフィルタサイズ
                rng=rng,
                height_scale=float(rng.uniform(0.9, 1.1)),   # 数字の大きさの変動
                thick_scale=float(rng.uniform(0.7, 1.5)),    # 線の太さの変動
            )
            # 全てのピクセルの明るさおよびコントラストの変動
            gain = rng.uniform(0.8, 1.1)
            bias = rng.uniform(-25, 15)
            img = np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        else:
            img = pr.render_panel(layout, img_px, panel_size, rng=rng)
        cells = pr.extract_cells(img, panel_size, cell_px)      # (9, 32, 32)
        xs.append(cells)
        # リストをバラして追加する。
        ys.extend(d - 1 for d in layout)                        # クラス0..8 = 数字1..9
    # (N, 32, 32) -> (N, 32, 32, 1)
    x = np.concatenate(xs, axis=0)[..., None].astype(np.float32) # チャンネルの次元を追加する。
    y = np.array(ys, dtype=np.int64)
    return x, y


def build_model(keras):
    return keras.Sequential([
        keras.layers.Input((32, 32, 1)), # 高さ、幅、チャンネル数
        keras.layers.Conv2D(8, 3, activation="relu"),    # 3x3、8枚 (30,30,8)
        keras.layers.MaxPooling2D(2),                    # 2x2 -> (15,15,:)
        keras.layers.Conv2D(16, 3, activation="relu"),   # 3x3、16枚 (13,13,16)
        keras.layers.MaxPooling2D(2),                    # 2x2 -> (6,6,:)
        keras.layers.Flatten(),                          # 6*6*16 = 576
        keras.layers.Dense(32, activation="relu"),       # 576 -> 32
        keras.layers.Dense(9),                           # ロジット（softmaxは損失側で）
    ])


def export_npz(model, path):
    conv1, _, conv2, _, _, fc1, fc2 = model.layers
    # 複数のNumPy配列を1つのアーカイブファイル(.npz)にまとめて保存する。
    np.savez(
        path,
        conv1_w=conv1.get_weights()[0], conv1_b=conv1.get_weights()[1],
        conv2_w=conv2.get_weights()[0], conv2_b=conv2.get_weights()[1],
        fc1_w=fc1.get_weights()[0], fc1_b=fc1.get_weights()[1],
        fc2_w=fc2.get_weights()[0], fc2_b=fc2.get_weights()[1],
    )


def main():
    # コマンドラインオプションの定義
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", type=int, default=2500, help="学習用パネル枚数（1枚=9マス）")
    ap.add_argument("--test-panels", type=int, default=300, help="テスト用パネル枚数（1枚=9マス）")
    ap.add_argument("--epochs", type=int, default=8, help="学習エポック数")
    ap.add_argument("--batch", type=int, default=128, help="ミニバッチサイズ")
    ap.add_argument("--seed", type=int, default=0, help="乱数の種の値")
    ap.add_argument("--out", default="digit_cnn.npz", help="出力する重みファイルの名前")
    # コマンドラインオプションの処理
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow import keras
    # TensorFlow内部の乱数生成器
    tf.random.set_seed(args.seed)
    # NumPy側の乱数生成器
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    # 学習用データセットの生成
    x_train, y_train = make_dataset(args.panels, rng)
    # テスト用データセットの生成。学習用とは異なる乱数の種を使用する。
    x_test, y_test = make_dataset(args.test_panels, np.random.default_rng(args.seed + 1))
    print(f"data: train {x_train.shape}, test {x_test.shape}  ({time.time() - t0:.1f}s)")

    model = build_model(keras)
    # 各層の出力形状とパラメタ数の一覧を表示する
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3), # 学習率0.001
        # SparseCategoricalCrossentropy: 各正解の値はスカラ整数値でよい
        # from_logits=Trueで、モデルの出力はSoftmax前の値（ロジット）であり、
        #   Softmaxは損失関数の内部で計算することを指定する。
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"], # 正解率を表示する。
    )
    # 学習する。
    # validation_split=0.1は、学習データの末尾10%を検証用に分ける（ホールドアウト法）
    model.fit(x_train, y_train, validation_split=0.1,
              epochs=args.epochs, batch_size=args.batch, verbose=2)
    # テスト用データセットで評価する。
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"test accuracy（別のseedで生成したデータ）: {acc * 100:.2f}%")

    # 重みをファイルargs.outに出力する。
    export_npz(model, args.out)
    print(f"saved {args.out}")

    # NumPy版の推論とTensorFlow版の推論とが一致するか確認する（ROS側はNumPy版を使う）。
    from digit_cnn_np import DigitCNN
    net = DigitCNN(args.out)
    # TensorFlow版CNNの出力
    ref = tf.nn.softmax(model.predict(x_test[:500], verbose=0)).numpy()
    # NumPy版CNNの出力
    mine = net.predict_proba(x_test[:500])
    print(f"NumPy版とTensorFlow版の確率の最大差: {np.abs(ref - mine).max():.2e}")
    print(f"NumPy版の正解率: {(mine.argmax(1) == y_test[:500]).mean() * 100:.2f}% （500マス）")


if __name__ == "__main__":
    main()
