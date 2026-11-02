#!/usr/bin/env python3
"""
digit_cnn_np.py -- 数字認識CNNの軽量推論部

train_cnn.pyが書き出したdigit_cnn.npzを読み込み、
  Conv(3x3) -> ReLU -> MaxPool(2) -> Conv(3x3) -> ReLU -> MaxPool(2) -> Flatten -> Dense -> ReLU -> Dense
を計算する。NumPyだけで動作するのでROS環境にはTensorFlow不要。

入力: (N, 32, 32)または(N, 32, 32, 1)のfloat32 (0..1)
出力: (N, 9)の確率。クラスkは数字k+1に対応する。
"""

import numpy as np


class DigitCNN:
    def __init__(self, path):
        d = np.load(path) # 重みファイルの読み込み
        self.w1, self.b1 = d["conv1_w"], d["conv1_b"]
        self.w2, self.b2 = d["conv2_w"], d["conv2_b"]
        self.w3, self.b3 = d["fc1_w"], d["fc1_b"]
        self.w4, self.b4 = d["fc2_w"], d["fc2_b"]

    @staticmethod
    def _conv2d(x, w, b):
        """valid（外側のパディングなし）畳み込み。
            x: (N,H,W,C)
            w: (kh,kw,C,F)
        """
        n, h, wd, c = x.shape # バッチの個数、高さ、幅、チャンネル数
        kh, kw, _, f = w.shape # 高さ、幅、チャンネル数、フィルターの個数
        oh, ow = h - kh + 1, wd - kw + 1 # 出力の高さおよび幅
        cols = np.empty((n, oh, ow, kh, kw, c), dtype=x.dtype)
        # 入力xをcolsに詰め替え
        for i in range(kh):
            for j in range(kw):
                cols[:, :, :, i, j, :] = x[:, i:i + oh, j:j + ow, :]
        # 畳み込み(im2col)。戻り値の形状は(n, oh, ow, f)。
        return cols.reshape(n, oh, ow, kh * kw * c) @ w.reshape(kh * kw * c, f) + b

    @staticmethod
    def _maxpool2(x):
        """xに最大値プーリング処理を行った結果(n, h2, w2, c)を返す。"""
        n, h, w, c = x.shape # バッチの個数、高さ、幅、チャンネル数
        h2, w2 = h // 2, w // 2
        x = x[:, :h2 * 2, :w2 * 2, :] # 要素数が奇数の場合、最後の1行（列）を削除する。
        # 戻り値の形状は(n, h2, w2, c)
        return x.reshape(n, h2, 2, w2, 2, c).max(axis=(2, 4))

    def logits(self, x):
        """xをCNNおよび全結合層に通した結果を返す。"""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 3: # チャンネルの次元がない場合
            x = x[..., None]
        x = np.maximum(self._conv2d(x, self.w1, self.b1), 0.0) # Conv2D+ReLU
        x = self._maxpool2(x)
        x = np.maximum(self._conv2d(x, self.w2, self.b2), 0.0) # Conv2D+ReLU
        x = self._maxpool2(x)
        x = x.reshape(x.shape[0], -1)          # Kerasと同じ(H, W, C)順のFlatten
        x = np.maximum(x @ self.w3 + self.b3, 0.0) # 全結合層+ReLU
        return x @ self.w4 + self.b4 # 全結合層。形状は(N, 9)

    def predict_proba(self, x):
        """xをCNN+全結合層に通した結果のSoftmax、つまり各数字の確率を返す。"""
        z = self.logits(x) # CNN+全結合層
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)
