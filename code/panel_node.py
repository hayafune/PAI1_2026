#!/usr/bin/env python3
"""
panel_node.py -- 数字1〜9をランダムに配置した仮想パネルを配信する

トピックで配信するもの:
  /panel_camera/image  sensor_msgs/Image        仮想カメラ画像(bgr8)。認識ノードはこれだけを見る
  /panel/truth         std_msgs/Int32MultiArray 正解の配置（長さ9、行優先）。採点用。
  /touch_markers       visualization_msgs/MarkerArray  RViz用のパネル板および数字

サービス:
  /panel/shuffle       std_srvs/Trigger         数字の配置を並べ替える

使い方:
  python3 panel_node.py
  python3 panel_node.py --ros-args -p seed:=0                  # 再現できる数字の配置
  python3 panel_node.py --ros-args -p auto_shuffle_sec:=8.0    # 8秒ごとに自動で数字を並べ替える。
  ros2 service call /panel/shuffle std_srvs/srv/Trigger        # 手動で数字を並べ替える。

注意:
  touch_one_point.pyもトピック/touch_markersにパネルを描くので、同時に動かすと表示がちらつく。同時には実行しないようにする。
"""

import random

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

import panel_render as pr


class PanelNode(Node):
    def __init__(self):
        super().__init__("panel_node")

        # パラメタおよびその既定値の宣言。次の指定で変更可能:
        #   --ros-args -p パラメタ:=値
        # RViz用マーカーの基準となる座標系
        self.declare_parameter("frame", "panda_link0")
        # ロボット座標上でのパネル板のx方向の基準位置
        self.declare_parameter("panel_x", 0.50)
        # ロボット座標上でのパネル板のy、z方向の中心位置
        self.declare_parameter("panel_cy", 0.0)
        self.declare_parameter("panel_cz", 0.50)
        # パネル板（周辺の余白は含まない）の1辺の長さ(m)
        self.declare_parameter("panel_size", 0.30)
        # パネル板全体（周辺の余白を含む）の1辺のピクセルの個数
        self.declare_parameter("img_px", 360)
        self.declare_parameter("seed", -1)               # -1: 毎回ランダム
        self.declare_parameter("auto_shuffle_sec", 0.0)  # 0: 自動並べ替えなし
        self.declare_parameter("noise_std", 0.0)
        self.declare_parameter("blur_ksize", 0)
        self.declare_parameter("jitter_px", 0.0)
        self.declare_parameter("rate_hz", 2.0)

        # パラメタの値の設定
        p = self.get_parameter
        self.frame = p("frame").value
        self.panel_x = float(p("panel_x").value)
        self.panel_cy = float(p("panel_cy").value)
        self.panel_cz = float(p("panel_cz").value)
        self.panel_size = float(p("panel_size").value)
        self.img_px = int(p("img_px").value)
        seed = int(p("seed").value)
        self.noise_std = float(p("noise_std").value)
        self.blur_ksize = int(p("blur_ksize").value)
        self.jitter_px = float(p("jitter_px").value)

        # 数字の並べ替え用
        self.py_rng = random.Random(seed if seed >= 0 else None)
        # data augmentation用
        self.np_rng = np.random.default_rng(seed if seed >= 0 else None)

        self.layout = []
        self.shuffle() # self.layoutに、ランダムな整数列(1..9)のリストを設定する。

        # トピック通信の生成
        self.img_pub = self.create_publisher(Image, "panel_camera/image", 10)
        self.truth_pub = self.create_publisher(Int32MultiArray, "panel/truth", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "touch_markers", 10)
        # サービス通信の生成
        self.create_service(Trigger, "panel/shuffle", self.on_shuffle)

        # トピック配信用
        self.create_timer(1.0 / float(p("rate_hz").value), self.publish_all)
        # 数字の並べ替え用
        auto = float(p("auto_shuffle_sec").value)
        if auto > 0:
            self.create_timer(auto, self.shuffle)

    # ------------------------------------------------------------------
    def shuffle(self):
        """self.layoutに整数列(1..9)をランダムに並べ替えたリストを設定する。"""
        layout = list(range(1, 10))
        self.py_rng.shuffle(layout)
        self.layout = layout
        rows = [layout[0:3], layout[3:6], layout[6:9]]
        self.get_logger().info(f"new layout: {rows[0]} / {rows[1]} / {rows[2]}")

    def on_shuffle(self, request, response):
        """self.layoutに整数列(1..9)をランダムに並べ替えたリストを設定し、それを文字列化したメッセージをサービス応答する。"""
        self.shuffle()
        response.success = True
        response.message = str(self.layout)
        return response

    # ------------------------------------------------------------------
    def cell_center(self, row, col):
        """マス(row, col)の中心位置のロボット座標を返す。
        左が+y、上が+z
        """
        pitch = self.panel_size / 3.0 # マス1辺の長さ(m)
        y = self.panel_cy + (1 - col) * pitch
        z = self.panel_cz + (1 - row) * pitch
        return self.panel_x, y, z

    def publish_all(self):
        """正解、仮想カメラ画像およびRViz用のマーカーをトピック配信する"""
        now = self.get_clock().now().to_msg()
        self.publish_truth()    # 正解を先に配信し、採点を容易にする。
        self.publish_image(now) # 仮想カメラ画像
        self.publish_markers()  # RViz用のマーカー

    def publish_image(self, stamp):
        """仮想カメラの画像をトピック配信する。"""
        img = pr.render_panel(
            self.layout, img_px=self.img_px, panel_size=self.panel_size,
            jitter_px=self.jitter_px, noise_std=self.noise_std,
            blur_ksize=self.blur_ksize, rng=self.np_rng,
        )
        # https://docs.ros2.org/foxy/api/sensor_msgs/msg/Image.html
        msg = Image() # 画像メッセージインスタンス
        msg.header.stamp = stamp
        msg.header.frame_id = "panel_camera" # 仮想カメラ
        msg.height, msg.width = img.shape[0], img.shape[1] # ピクセル
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = img.shape[1] * 3 # 3色分のバイト数
        msg.data = img.tobytes() # NumPy to バイト列
        self.img_pub.publish(msg)

    def publish_truth(self):
        """正解の数字の並びをトピック配信する。"""
        # https://docs.ros2.org/foxy/api/std_msgs/msg/Int32MultiArray.html
        msg = Int32MultiArray()
        msg.data = [int(d) for d in self.layout]
        self.truth_pub.publish(msg)

    def _marker(self, mid, mtype):
        """RViz用のマーカーを生成し返す。"""
        # https://docs.ros2.org/galactic/api/visualization_msgs/msg/Marker.html
        m = Marker()
        m.header.frame_id = self.frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "touch_demo" # RViz内でのグループ名
        m.id = mid # マーカーのID
        m.type = mtype # マーカーの形
        m.action = Marker.ADD # 追加する。
        m.pose.orientation.w = 1.0 # 既定値の姿勢（回転なし）
        return m

    def publish_markers(self):
        """パネル板+数字に対応するRViz用マーカーを生成しトピック配信する"""
        # https://docs.ros2.org/foxy/api/visualization_msgs/msg/MarkerArray.html
        arr = MarkerArray()

        # パネル板に対応するマーカーの生成
        plate = self._marker(0, Marker.CUBE) # 立方体
        # 中心位置のロボット座標
        plate.pose.position.x = self.panel_x + 0.006
        plate.pose.position.y = self.panel_cy
        plate.pose.position.z = self.panel_cz
        plate.scale.x = 0.01 # パネル板の厚み(m): (self.panel_x+0.001:self.panel_x+0.011)
        # パネル板全体（外側の余白を含む）の一辺の長さ(m)を設定する。
        plate.scale.y = pr.plate_size(self.panel_size)
        plate.scale.z = pr.plate_size(self.panel_size)
        plate.color.r = plate.color.g = plate.color.b = 0.85 # 明るい灰色
        plate.color.a = 1.0 # 不透明（半透明だと、見る角度によって後ろのものが透けて見える）
        arr.markers.append(plate)

        # 1..9の数字に対応するマーカーの生成
        for i, digit in enumerate(self.layout):
            row, col = divmod(i, 3) # linear to 2D (row-major)
            x, y, z = self.cell_center(row, col)
            # 常に文字がこちら側を向く文字マーカー
            t = self._marker(100 + i, Marker.TEXT_VIEW_FACING)
            t.pose.position.x = x - 0.012
            t.pose.position.y = y
            t.pose.position.z = z
            t.scale.z = 0.06 # 文字の高さ(m)
            t.color.r = t.color.g = t.color.b = 0.05 # ほぼ黒色
            t.color.a = 1.0 # 不透明
            t.text = str(digit)
            arr.markers.append(t)

        self.marker_pub.publish(arr)


def main():
    rclpy.init()
    node = PanelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
