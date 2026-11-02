#!/usr/bin/env python3
"""
image_viewer.py -- sensor_msgs/ImageをOpenCVのウィンドウに表示する簡易ビューア

RVizのImageディスプレイがWSL2で落ちる場合の代わり。

使い方:
  python3 image_viewer.py
  python3 image_viewer.py --ros-args -p topic:=panel_camera/image -p scale:=1.5
  （終了はウィンドウ上でqキーまたはCtrl+C）

cv2.imshowが使えない環境では、代わりに次を使う:
  sudo apt install ros-jazzy-rqt-image-view
  ros2 run rqt_image_view rqt_image_view
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class ImageViewer(Node):
    def __init__(self):
        super().__init__("image_viewer")

        # パラメタおよびその既定値の宣言。次の指定で変更可能:
        #   --ros-args -p パラメタ:=値
        self.declare_parameter("topic", "panel_camera/image")
        self.declare_parameter("scale", 1.0)

        # パラメタの値の設定
        topic = self.get_parameter("topic").value
        self.scale = float(self.get_parameter("scale").value)
        self.title = f"image_viewer: {topic}"
        self.latest = None

        # トピック通信（購読）
        self.create_subscription(Image, topic, self.on_image, 10)
        self.create_timer(0.05, self.show)  # 20 Hzで描画
        self.get_logger().info(f"subscribing to {topic}")

    def on_image(self, msg):
        """配信された画像をNumPy配列に変換してself.latestに設定する。"""
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding in ("bgr8", "rgb8"): # カラーの場合
            img = arr.reshape(msg.height, msg.width, 3)
            if msg.encoding == "rgb8": # BGRに変換
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif msg.encoding == "mono8": # gray-scaleの場合、BGRに変換
            img = cv2.cvtColor(arr.reshape(msg.height, msg.width), cv2.COLOR_GRAY2BGR)
        else: # 用意されていないencoding
            self.get_logger().warn(f"unsupported encoding: {msg.encoding}", throttle_duration_sec=5.0)
            return
        self.latest = img

    def show(self):
        """最新の画像をウィンドウに描画する。キーqが押されたら終了する。"""
        if self.latest is None:
            return
        img = self.latest
        if self.scale != 1.0: # 画像をself.scale倍する。
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        # 静止画imgの描画をOSに依頼する。
        cv2.imshow(self.title, img)
        # cv2.waitKey(1)でGUIの描画処理を起動し、最大1 msだけOSとのイベント処理を行う
        # waitKeyの戻り値（キーコード、キー入力がない場合は-1）の下位1 Byteが"q"なら
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt


def main():
    rclpy.init()
    node = ImageViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # 関数show中のKeyboardInterruptを受け取り正常終了に進む
        pass
    finally:
        cv2.destroyAllWindows() # CV2が開いた全てのウィンドウを閉じる。
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
