#!/usr/bin/env python3
"""
image_viewer.py -- sensor_msgs/Image を OpenCV のウィンドウに表示する簡易ビューア

RViz の Image ディスプレイが WSL2 で落ちる場合の代わり。

使い方:
  python3 image_viewer.py
  python3 image_viewer.py --ros-args -p topic:=panel_camera/image -p scale:=1.5
  (終了はウィンドウ上で q キー、または Ctrl+C)

cv2.imshow が使えない環境では、代わりに次を使う:
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
        self.declare_parameter("topic", "panel_camera/image")
        self.declare_parameter("scale", 1.0)
        topic = self.get_parameter("topic").value
        self.scale = float(self.get_parameter("scale").value)
        self.title = f"image_viewer: {topic}"
        self.latest = None

        self.create_subscription(Image, topic, self.on_image, 10)
        self.create_timer(0.05, self.show)  # 20 Hz で描画
        self.get_logger().info(f"subscribing to {topic}")

    def on_image(self, msg):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding in ("bgr8", "rgb8"):
            img = arr.reshape(msg.height, msg.width, 3)
            if msg.encoding == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif msg.encoding == "mono8":
            img = cv2.cvtColor(arr.reshape(msg.height, msg.width), cv2.COLOR_GRAY2BGR)
        else:
            self.get_logger().warn(f"unsupported encoding: {msg.encoding}", throttle_duration_sec=5.0)
            return
        self.latest = img

    def show(self):
        if self.latest is None:
            return
        img = self.latest
        if self.scale != 1.0:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        cv2.imshow(self.title, img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt


def main():
    rclpy.init()
    node = ImageViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
