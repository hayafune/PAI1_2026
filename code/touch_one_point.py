#!/usr/bin/env python3
"""
touch_one_point.py -- Step 1: 仮想パネルの1マス(または9マス)を指先でタッチする

前提:
  - MoveIt 2 (Fake Hardware) が Panda 構成で起動済み (RViz 付き)
  - pymoveit2 が import できる (ワークスペースを source 済み)

使い方:
  python3 touch_one_point.py                                   # 中央マス(cell=4)だけタッチ
  python3 touch_one_point.py --ros-args -p cell:=0             # 左上マス
  python3 touch_one_point.py --ros-args -p mode:=grid          # 9マスを順番にタッチ(届くか確認)

座標系: Panda のベースリンク (panda_link0)。
  +x = ロボットの前方, +y = ロボットの左, +z = 上
  パネルは x = panel_x の鉛直面。ロボット側から見て3x3 のマス。
  マス番号0..8 は行優先(左上=0, 右上=2, 左下=6, 右下=8)。表示する数字は番号+1。

RViz 側の設定:
  Add -> MarkerArray -> Topic = /touch_markers, Fixed Frame = panda_link0
"""

import math
import time
from threading import Thread

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from pymoveit2 import MoveIt2
from pymoveit2.robots import panda

# ハンドの z 軸(指の向き)を +x(前方)へ向ける: y軸まわり +90度
QUAT_FORWARD = [0.0, 0.7071068, 0.0, 0.7071068]

# Panda の標準的な待機姿勢 (rad)
READY = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]


class TouchDemo(Node):
    def __init__(self):
        super().__init__("touch_demo")

        # ---- パラメタ ----
        self.declare_parameter("mode", "single")      # single | grid
        self.declare_parameter("cell", 4)             # single のときのマス番号0..8
        self.declare_parameter("panel_x", 0.50)       # パネル面の x [m]
        self.declare_parameter("panel_cy", 0.0)       # パネル中心の y [m]
        self.declare_parameter("panel_cz", 0.50)      # パネル中心の z [m]
        self.declare_parameter("panel_size", 0.30)    # パネル一辺 [m] (3x3 なので1マスは size/3)
        self.declare_parameter("tcp_offset", 0.10)    # panda_hand 原点から指先までの距離 [m]
        self.declare_parameter("pre_touch", 0.05)     # タッチ前に止まる手前距離 [m]
        self.declare_parameter("velocity", 0.3)       # 速度・加速度スケール0..1

        p = self.get_parameter
        self.mode = p("mode").value
        self.cell = int(p("cell").value)
        self.panel_x = float(p("panel_x").value)
        self.panel_cy = float(p("panel_cy").value)
        self.panel_cz = float(p("panel_cz").value)
        self.panel_size = float(p("panel_size").value)
        self.tcp_offset = float(p("tcp_offset").value)
        self.pre_touch = float(p("pre_touch").value)
        vel = float(p("velocity").value)

        self.base_frame = panda.base_link_name()

        # ---- MoveIt 2 (pymoveit2) ----
        moveit_kwargs = dict(
            node=self,
            joint_names=panda.joint_names(),
            base_link_name=self.base_frame,
            # pymoveit2 同梱の panda 定義 (group="arm", ee="panda_hand_tcp") は
            # moveit_resources の Panda 設定と名前が違うため、ここで直接指定する
            end_effector_name="panda_hand",
            group_name="panda_arm",
            callback_group=ReentrantCallbackGroup(),
        )
        try:
            # moveit_resources の Panda はコントローラ名が panda_arm_controller のため、
            # 軌道の実行は move_group (ExecuteTrajectory) 経由にして名前の違いを避ける
            self.moveit2 = MoveIt2(**moveit_kwargs, execute_via_moveit=True)
        except TypeError:
            # 古い pymoveit2 でこの引数が無い場合
            self.moveit2 = MoveIt2(**moveit_kwargs)
        self.moveit2.max_velocity = vel
        self.moveit2.max_acceleration = vel

        # ---- RViz 用マーカー (パネル・数字・ターゲット) ----
        self.marker_pub = self.create_publisher(MarkerArray, "touch_markers", 10)
        self.target_tip = None
        self.create_timer(1.0, self.publish_markers)

    # ------------------------------------------------------------------
    # 幾何
    # ------------------------------------------------------------------
    def cell_center(self, row, col):
        """マス(row, col)の中心 (x, y, z)。row=0 が上段、col=0 が左(ロボット視点で +y 側)。"""
        pitch = self.panel_size / 3.0
        y = self.panel_cy + (1 - col) * pitch
        z = self.panel_cz + (1 - row) * pitch
        return self.panel_x, y, z

    # ------------------------------------------------------------------
    # 動作
    # ------------------------------------------------------------------
    def go(self, tip_x, tip_y, tip_z, cartesian):
        """指先を (tip_x, tip_y, tip_z) に動かす。MoveIt にはハンド原点の目標を渡す。"""
        hand_x = tip_x - self.tcp_offset  # 手先は +x を向いているので、ハンド原点は指先より後ろ
        self.moveit2.move_to_pose(
            position=[hand_x, tip_y, tip_z],
            quat_xyzw=QUAT_FORWARD,
            cartesian=cartesian,
        )
        ok = self.moveit2.wait_until_executed()
        if ok is False:
            self.get_logger().warn(
                f"move failed: tip=({tip_x:.3f},{tip_y:.3f},{tip_z:.3f}) cartesian={cartesian}"
            )
            return False
        return True

    def touch_cell(self, index):
        row, col = divmod(index, 3)
        x, y, z = self.cell_center(row, col)
        self.target_tip = (x, y, z)
        self.get_logger().info(f"touch cell {index} (digit {index + 1}) tip=({x:.3f},{y:.3f},{z:.3f})")

        if not self.go(x - self.pre_touch, y, z, cartesian=False):  # 手前へ(通常のプランニング)
            return False
        if not self.go(x, y, z, cartesian=True):                    # 直線で前進 = タッチ
            return False
        time.sleep(0.3)
        self.report_tip_error((x, y, z))
        return self.go(x - self.pre_touch, y, z, cartesian=True)    # 直線で後退

    def report_tip_error(self, target):
        """FK で指先位置を取り、目標との誤差をログ出力 (Step 4 の成功判定の予行演習)。"""
        try:
            fk = self.moveit2.compute_fk()
            if isinstance(fk, list):
                fk = fk[0]
            q = fk.pose.position
            tip = (q.x + self.tcp_offset, q.y, q.z)
            err = math.dist(tip, target)
            self.get_logger().info(f"  touch error = {err * 1000:.1f} mm")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"  FK check skipped: {e}")

    def run(self):
        self.get_logger().info("move to ready pose")
        self.moveit2.move_to_configuration(READY)
        self.moveit2.wait_until_executed()

        cells = range(9) if self.mode == "grid" else [self.cell]
        results = {}
        for i in cells:
            results[i] = self.touch_cell(i)

        self.get_logger().info("back to ready pose")
        self.moveit2.move_to_configuration(READY)
        self.moveit2.wait_until_executed()

        ng = [i for i, ok in results.items() if not ok]
        self.get_logger().info(f"done. success={len(results) - len(ng)}/{len(results)} failed cells={ng}")

    # ------------------------------------------------------------------
    # RViz マーカー
    # ------------------------------------------------------------------
    def _base_marker(self, mid, mtype):
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "touch_demo"
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        return m

    def publish_markers(self):
        arr = MarkerArray()

        # パネル本体(薄い板)
        panel = self._base_marker(0, Marker.CUBE)
        panel.pose.position.x = self.panel_x + 0.006
        panel.pose.position.y = self.panel_cy
        panel.pose.position.z = self.panel_cz
        panel.scale.x = 0.01
        panel.scale.y = self.panel_size + 0.04
        panel.scale.z = self.panel_size + 0.04
        panel.color.r = panel.color.g = panel.color.b = 0.85
        panel.color.a = 0.9
        arr.markers.append(panel)

        # 各マスの数字(Step 2 以降は認識対象の数字に差し替える)
        for i in range(9):
            row, col = divmod(i, 3)
            x, y, z = self.cell_center(row, col)
            t = self._base_marker(100 + i, Marker.TEXT_VIEW_FACING)
            t.pose.position.x = x - 0.012
            t.pose.position.y = y
            t.pose.position.z = z
            t.scale.z = 0.06
            t.color.r = t.color.g = t.color.b = 0.05
            t.color.a = 1.0
            t.text = str(i + 1)
            arr.markers.append(t)

        # 現在のターゲット(赤い球)
        if self.target_tip is not None:
            s = self._base_marker(200, Marker.SPHERE)
            s.pose.position.x, s.pose.position.y, s.pose.position.z = self.target_tip
            s.scale.x = s.scale.y = s.scale.z = 0.02
            s.color.r = 1.0
            s.color.a = 1.0
            arr.markers.append(s)

        self.marker_pub.publish(arr)


def main():
    rclpy.init()
    node = TouchDemo()

    executor = MultiThreadedExecutor(2)
    executor.add_node(node)
    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    node.create_rate(1.0).sleep()  # joint_states などが届くのを少し待つ

    try:
        node.run()
    finally:
        rclpy.shutdown()
        spin_thread.join()


if __name__ == "__main__":
    main()
