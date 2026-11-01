#!/usr/bin/env python3
"""
perception_node.py -- 仮想カメラ画像から数字を認識して、位置付きで配信する。

入力トピック:
  /panel_camera/image   sensor_msgs/Image         panel_node.pyが配信する仮想カメラ画像
  /panel/truth          std_msgs/Int32MultiArray  正解の配置。採点（ログ）にだけ使い、認識には使わない

出力トピック:
  /perception/digits      std_msgs/Int32MultiArray    認識した数字列（長さ9、マス番号の順）
  /perception/confidence  std_msgs/Float32MultiArray  各マスの認識結果の確信度 (0..1)
  /perception/points      geometry_msgs/PoseArray     各マスの中央の位置のロボット座標（原点: ロボットのベースリンク(panda_link0)の位置。+x: ロボットの前方、+y: 左、+z: 上）

認識の手順:
  1. 画像から9マスを切り出す(panel_render.extract_cells)
  2. CNN(digit_cnn_np.DigitCNN)で各マスの数字の確率を出す
  3. 「1〜9が1回ずつ現れる」という制約条件を利用し、全体の確率が最大になる並びを選ぶ(unique_decode)
    → CNNが間違えたマスがあっても、重複（同じ数字が複数か所）を避けて正しく直せる場合がある

使い方:
  python3 perception_node.py                           # 重みファイルdigit_cnn.npzを同じフォルダから読む
  python3 perception_node.py --ros-args -p unique_decode:=false   # 制約条件を利用せず
"""

import itertools
import os

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int32MultiArray

import panel_render as pr
from digit_cnn_np import DigitCNN


def build_permutations():
    """0..8の全ての順列（9! = 362880通り）を形状(362880, 9)の配列にする。"""
    return np.array(list(itertools.permutations(range(9))), dtype=np.int8)


def decode_unique(probs, perms):
    """
    probs: (9, 9)  probs[cell, k] = マスcellが数字k+1である確率
    perms: build_permutations()の結果
    戻り値: (9,)  各マスのクラス(0..8)。1〜9が1回ずつ現れる並びのうち、対数確率の和が最大のもの。
    """
    logp = np.log(np.clip(probs, 1e-9, 1.0))
    # 1次元目の添字であるマス番号配列[0,1,...,8]を形状(1, 9)とし、
    # 2次元目の添字であるpermsの形状(362880, 9)にbroadcast。
    scores = logp[np.arange(9)[None, :], perms].sum(axis=1) # (362880, 9)->(362880)
    return perms[int(np.argmax(scores))].astype(np.int64) # permsの行番号指定


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        here = os.path.dirname(os.path.abspath(__file__))
        # パラメタおよびその既定値の宣言。次の指定で変更可能:
        #   --ros-args -p パラメタ:=値
        # CNNの重みファイルの名前
        self.declare_parameter("model", os.path.join(here, "digit_cnn.npz"))
        # RViz用マーカーの基準となる座標系（panel_node.pyと同一）
        self.declare_parameter("frame", "panda_link0")
        # ロボット座標上でのパネル板のx方向の基準位置（panel_node.pyと同一）
        self.declare_parameter("panel_x", 0.50)
        # ロボット座標上でのパネル板のy、z方向の中心位置（panel_node.pyと同一）
        self.declare_parameter("panel_cy", 0.0)
        self.declare_parameter("panel_cz", 0.50)
        # パネル板（周辺の余白は含まない）の1辺の長さ(m)（panel_node.pyと同一）
        self.declare_parameter("panel_size", 0.30)
        # 制約条件を利用するか否か
        self.declare_parameter("unique_decode", True)
        self.declare_parameter("log_every", 20) # 何フレームごとに正解率をログに出すか
        # パラメタの値の設定
        p = self.get_parameter
        self.frame = p("frame").value
        self.panel_x = float(p("panel_x").value)
        self.panel_cy = float(p("panel_cy").value)
        self.panel_cz = float(p("panel_cz").value)
        self.panel_size = float(p("panel_size").value)
        self.unique_decode = bool(p("unique_decode").value)
        self.log_every = int(p("log_every").value)

        model_path = p("model").value
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{model_path}が見つかりません。先にtrain_cnn.pyで学習して重みファイルdigit_cnn.npzを作ってください。"
            )
        self.net = DigitCNN(model_path)
        self.perms = build_permutations() if self.unique_decode else None
        self.get_logger().info(f"model loaded: {model_path}  unique_decode={self.unique_decode}")

        # 初期化
        self.truth = None
        self.n_frames = 0
        # CNNによる認識結果において、マスの数値が正解であった総数
        self.n_cell_ok_raw = 0
        # 最終的な認識結果において、マスの数値が正解であった総数
        self.n_cell_ok_dec = 0
        # CNNによる認識結果において、各パネルの全ての数字が正解であった総数
        self.n_panel_ok_raw = 0
        # 最終的な認識結果において、各パネルの全ての数字が正解であった総数
        self.n_panel_ok_dec = 0
        # 採点したパネルの個数
        self.n_scored = 0
        self.last_truth_layout = None

        # トピックの購読
        # 仮想カメラ画像
        self.create_subscription(Image, "panel_camera/image", self.on_image, 10)
        # 正解
        self.create_subscription(Int32MultiArray, "panel/truth", self.on_truth, 10)
        # トピックの配信
        # 各マスの認識結果の数字のリスト
        self.digits_pub = self.create_publisher(Int32MultiArray, "perception/digits", 10)
        # 各マスの認識結果の確信度のリスト
        self.conf_pub = self.create_publisher(Float32MultiArray, "perception/confidence", 10)
        # 3D位置の配列
        self.points_pub = self.create_publisher(PoseArray, "perception/points", 10)

    # ------------------------------------------------------------------
    def on_truth(self, msg):
        """self.truthに、正解の数字の並びのリストを設定する"""
        self.truth = [int(v) for v in msg.data]

    def on_image(self, msg):
        if msg.encoding != "bgr8":
            self.get_logger().warn(f"unsupported encoding: {msg.encoding}", throttle_duration_sec=5.0)
            return
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        # 各マスの画像の切り出し
        cells = pr.extract_cells(img, self.panel_size)            # (9, 32, 32)
        # 各マスの数字の確率
        probs = self.net.predict_proba(cells)                      # (9, 9)

        # CNNだけによる各マスの数字の認識結果
        raw = probs.argmax(axis=1)
        # 最終的な各マスの数字の認識結果。unique_decodeの値が真なら制約条件を利用する
        dec = decode_unique(probs, self.perms) if self.unique_decode else raw
        # 最終的な各マスの数字の認識結果のリスト
        digits = (dec + 1).astype(int).tolist()
        # 各数字の確信度のリスト
        conf = probs[np.arange(9), dec].astype(float).tolist()

        self.publish(msg.header.stamp, digits, conf, img.shape[0])
        self.score(raw + 1, np.array(digits))

    # ------------------------------------------------------------------
    def publish(self, stamp, digits, conf, img_px):
        # 各マスの認識結果の数字のリストをトピック配信する。
        # https://docs.ros2.org/foxy/api/std_msgs/msg/Int32MultiArray.html
        d = Int32MultiArray()
        d.data = digits
        self.digits_pub.publish(d)

        # 各マスの認識結果の確信度のリストをトピック配信する。
        # https://docs.ros2.org/galactic/api/std_msgs/msg/Float32MultiArray.html
        c = Float32MultiArray()
        c.data = [float(v) for v in conf]
        self.conf_pub.publish(c)

        # 各マスの中心位置のロボット座標をトピック配信する。
        # https://docs.ros2.org/foxy/api/geometry_msgs/msg/PoseArray.html
        poses = PoseArray()
        poses.header.stamp = stamp
        poses.header.frame_id = self.frame
        for idx in range(9):
            row, col = divmod(idx, 3) # linear to 2D (row-major)
            # マス(row, col)の中心のピクセル座標(u, v)
            u, v = pr.cell_center_px(row, col, img_px, self.panel_size)
            # ピクセル座標(u, v) to パネル座標(dy, dz)。
            dy, dz = pr.pixel_to_panel(u, v, img_px, self.panel_size)
            # https://docs.ros2.org/foxy/api/geometry_msgs/msg/Pose.html
            pose = Pose()
            pose.position.x = self.panel_x
            pose.position.y = self.panel_cy + dy
            pose.position.z = self.panel_cz + dz
            pose.orientation.w = 1.0 # 正面を真っ直ぐ向いた既定値の姿勢（回転なし）
            poses.poses.append(pose)
        self.points_pub.publish(poses)

    def score(self, raw_digits, dec_digits):
        """正解と照合して、マス単位・パネル単位の正解率をログに出力する。"""
        self.n_frames += 1 # 処理したパネルの総数（使用せず）
        if self.truth is None or len(self.truth) != 9:
            return # 正解が未配信
        truth = np.array(self.truth)

        # 並べ替え直後は正解と画像とが1フレームずれる可能性があるので、正解が変わった直後は数えない
        if self.last_truth_layout != self.truth:
            self.last_truth_layout = list(self.truth)
            return

        self.n_scored += 1
        # マス単位の正解の総数
        self.n_cell_ok_raw += int((raw_digits == truth).sum())
        self.n_cell_ok_dec += int((dec_digits == truth).sum())
        # パネル単位の正解の総数
        self.n_panel_ok_raw += int((raw_digits == truth).all())
        self.n_panel_ok_dec += int((dec_digits == truth).all())

        if self.n_scored % self.log_every == 0:
            n = self.n_scored
            self.get_logger().info(
                f"frames={n}  "
                # マス単位の正解率
                f"cell acc: raw {100 * self.n_cell_ok_raw / (9 * n):.1f}% / decoded {100 * self.n_cell_ok_dec / (9 * n):.1f}%  "
                # パネル単位の正解率
                f"panel acc: raw {100 * self.n_panel_ok_raw / n:.1f}% / decoded {100 * self.n_panel_ok_dec / n:.1f}%"
            )


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
