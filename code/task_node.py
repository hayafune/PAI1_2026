#!/usr/bin/env python3
"""
task_node.py -- 認識した数字を小さい順(1→9)にタッチして、採点・記録する

1試行の流れ:
  1. サービス通信/panel/shuffleで数字の配置を並べ替える
  2. 認識結果(/perception/digits, /perception/points)が3回連続で同じになるまで待つ
  3. 認識した数字の小さい順に、各マスの位置へ「手前→前進（タッチ）→後退」で指先を動かす
  4. 指先の実際の位置（順運動学）からタッチしたマスを求め、正解(/panel/truth)を参照して
     「n番目にタッチした数字がnか否か」を採点する。正解はここでだけ使う。
  5. 間違えたら試行を中断する（abort_on_miss:=trueの場合）。9個全て正しければ成功

事前準備（別々のターミナルで）:
  ros2 launch moveit_resources_panda_moveit_config demo.launch.py
  python3 panel_node.py           # auto_shuffle_secは付けない（試行の途中で並べ替わらないように）
  python3 perception_node.py
  python3 image_viewer.py         # 動画撮影用

実行:
  python3 task_node.py                                    # 1試行
  python3 task_node.py --ros-args -p trials:=10           # 10試行して成功率を出す
  python3 task_node.py --ros-args -p trials:=10 -p velocity:=0.8   # 速く動かす
  python3 task_node.py --ros-args -p stable_frames:=1              # 3フレームの合意を使わない（劣化した画像での比較用）
  python3 task_node.py --ros-args -p status_size:=0.07 -p label:=clean   # 動画用: 進行表示を大きく、条件名clean（白文字・太字ふう）

結果は既定値ではファイルresults.csvに追記される。
"""

import csv
import math
import os
import time
from collections import deque
from threading import Thread

import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from pymoveit2 import MoveIt2
from pymoveit2.robots import panda

# ロボットアーム(Panda)の手の向き（姿勢）
#  手先がまっすぐパネル方向を向く: y軸周りπ/2:  四元数[0, 2^(-1/2), 0, 2^(-1/2)]
QUAT_FORWARD = [0.0, 0.7071068, 0.0, 0.7071068]
# 待機姿勢時の7つの関節の角度（readyポーズ）
#  https://moveit.picknik.ai/main/doc/examples/setup_assistant/setup_assistant_tutorial.html
READY = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]


# ----------------------------------------------------------------------
# ROSに依存しない関数
# ----------------------------------------------------------------------
def plan_order(digits):
    """認識した数字の小さい順にマス番号を並べたリストを返す。
       同じ数字があったらマス番号の小さい順に並べる（安定ソート）。

       digits: 認識した数字の列（マスの行優先順）
    """
    return sorted(range(len(digits)), key=lambda i: (digits[i], i))


def tip_to_cell(y, z, cy, cz, panel_size):
    """指先のロボット座標(y, z)からタッチしたマス番号（行優先0..8）を返す。パネルの外ならNoneを返す。
        cy, cz: パネル中心のロボット座標。左が+y、上が+z
    """
    pitch = panel_size / 3.0 # マス1辺の長さ(m)
    # roundは偶数丸め。ロボットはマスの中心近くをタッチする。
    col = 1 - int(round((y - cy) / pitch))   # 左 (+y) が col 0
    row = 1 - int(round((z - cz) / pitch))   # 上 (+z) が row 0
    if 0 <= row <= 2 and 0 <= col <= 2:
        return row * 3 + col
    return None


# ----------------------------------------------------------------------
class TaskNode(Node):
    def __init__(self):
        super().__init__("task_node")

        # パラメタおよびその既定値の宣言。次の指定で変更可能:
        #   --ros-args -p パラメタ:=値
        self.declare_parameter("trials", 1) # タスクを実行する回数
        self.declare_parameter("abort_on_miss", True)
        # ロボット座標上でのパネル板のx方向の基準位置（panel_node.pyと同一）
        self.declare_parameter("panel_x", 0.50)
        # ロボット座標上でのパネル板のy、z方向の中心位置（panel_node.pyと同一）
        self.declare_parameter("panel_cy", 0.0)
        self.declare_parameter("panel_cz", 0.50)
        # パネル板（周辺の余白は含まない）の1辺の長さ(m)（panel_node.pyと同一）
        self.declare_parameter("panel_size", 0.30)
        self.declare_parameter("tcp_offset", 0.10) # ハンドの原点から指先までの距離(m)
        # タッチするとき、まずマスのx座標pre_touch m手前に移動する。
        self.declare_parameter("pre_touch", 0.05)
        self.declare_parameter("velocity", 0.5) # アームの移動速度の倍率(0~1)
        self.declare_parameter("touch_tol", 0.02)    # 指先のx座標の値がパネル面からこの範囲内なら「タッチした」
        self.declare_parameter("status_dz", 0.0)     # 進行表示を上下に動かす量(m)（上が正）。パネルと重なるときに調整する。
        self.declare_parameter("status_bold", True)  # 進行表示を太字ふうにするか否か（少しずらして重ね描きする）
        self.declare_parameter("status_bg", False)   # 進行表示の背景に暗い板を置くか否か。正面からの視点専用（斜めからだと大きな斜めの板に見える）
        self.declare_parameter("status_size", 0.05)  # 進行表示の文字の高さ(m)
        self.declare_parameter("label", "")          # 進行表示の先頭に付ける条件名（半角英数字。例: severe-raw）
        self.declare_parameter("stable_frames", 3)   # 何回連続で同じ認識結果なら採用するか
        self.declare_parameter("csv", "results.csv") # 出力するファイルの名前
        # RViz用マーカーの基準となる座標系
        self.declare_parameter("frame", panda.base_link_name())

        # パラメタの値の設定
        p = self.get_parameter
        self.trials = int(p("trials").value)
        self.abort_on_miss = bool(p("abort_on_miss").value)
        self.panel_x = float(p("panel_x").value)
        self.panel_cy = float(p("panel_cy").value)
        self.panel_cz = float(p("panel_cz").value)
        self.panel_size = float(p("panel_size").value)
        self.tcp_offset = float(p("tcp_offset").value)
        self.pre_touch = float(p("pre_touch").value)
        self.touch_tol = float(p("touch_tol").value)
        self.stable_frames = int(p("stable_frames").value)
        self.status_size = float(p("status_size").value)
        self.status_dz = float(p("status_dz").value)
        self.status_bold = bool(p("status_bold").value)
        self.status_bg = bool(p("status_bg").value)
        self.label = str(p("label").value)
        self.csv_path = p("csv").value
        self.frame = p("frame").value
        vel = float(p("velocity").value)

        # ---- 認識結果・正解の受信 ----
        self.recent = deque(maxlen=10)  # 直近10回の認識結果用のdequeを用意
        self.msg_count = 0   # トピックの受信回数
        self.points = None   # 行優先順の各マスの中心位置のロボット座標
        self.truth = None    # 正解
        # トピック通信（購読）
        self.create_subscription(Int32MultiArray, "perception/digits", self.on_digits, 10)
        self.create_subscription(PoseArray, "perception/points", self.on_points, 10)
        self.create_subscription(Int32MultiArray, "panel/truth", self.on_truth, 10)
        # サービス通信（クライアント）
        self.shuffle_cli = self.create_client(Trigger, "panel/shuffle")

        # トピック通信（発行）
        # RViz表示（パネルノードとは別の名前空間"task"を使い互いに消さないようにする）
        self.marker_pub = self.create_publisher(MarkerArray, "touch_markers", 10)
        # ---- MoveIt 2 (pymoveit2) ----
        kwargs = dict(
            node=self, # このノード
            joint_names=panda.joint_names(),
            base_link_name=panda.base_link_name(), # panda_link0
            end_effector_name="panda_hand",   # moveit_resourcesのPanda設定に合わせる
            group_name="panda_arm", # 動作対象とする関節のグループ名
            callback_group=ReentrantCallbackGroup(), # ROS2のマルチスレッド通信設定
        )
        try:
            self.moveit2 = MoveIt2(**kwargs, execute_via_moveit=True)
        except TypeError: # pymoveit2が引数execute_via_moveitに未対応の版の場合
            self.moveit2 = MoveIt2(**kwargs)
        self.moveit2.max_velocity = vel
        self.moveit2.max_acceleration = vel

    # ------------------------------------------------------------------
    # 受信
    # ------------------------------------------------------------------
    def on_digits(self, msg):
        """self.recentに認識した数字列を加え、トピックの受信回数を1だけ加算する。"""
        self.recent.append(tuple(int(v) for v in msg.data))
        self.msg_count += 1

    def on_points(self, msg):
        """self.pointsに行優先順で各マスの中心位置のロボット座標のリストを設定する。"""
        self.points = [(q.position.x, q.position.y, q.position.z) for q in msg.poses]

    def on_truth(self, msg):
        """self.truthに正解の数字列のリストを設定する。"""
        self.truth = [int(v) for v in msg.data]

    def wait_stable(self, count_before, need, timeout=15.0):
        """
        数字の並び替え後に届いた認識結果を待つ。
          need回連続で同じ結果になれば(digits, points, True)を返す。
          timeoutまでに揃わなければ、最新の結果を(digits, points, False)で返す（試行は続ける）。
          ここで、digitsは認識した数字列、pointsは行優先順のマスの中心のロボット座標。
        並び替え直後に届く画像が並び替え前のものである可能性を考慮し、最低2件は新しい結果を待つ。
            count_before: 数字の並べ替え直前のトピックの受信回数
        """
        need = max(1, need)
        t0 = time.time()
        while time.time() - t0 < timeout:
            # 最低2回はチェックする。
            if self.msg_count - count_before >= max(need, 2) and self.points:
                last = list(self.recent)[-need:] # 最近need個の認識結果
                if len(last) == need and len(set(last)) == 1:
                    return list(last[-1]), list(self.points), True
            time.sleep(0.1)
        if self.msg_count - count_before >= 2 and self.recent and self.points:
            return list(self.recent[-1]), list(self.points), False
        # トピックの受信がないなど、動作がおかしい場合
        return None

    def shuffle(self, timeout=5.0):
        """サービス通信/panel/shuffleを使用して数字を並び替える。
           戻り値: 並び替えが成功した場合True。失敗した場合False。
        """
        if not self.shuffle_cli.wait_for_service(timeout_sec=timeout):
            self.get_logger().error("/panel/shuffleが見つかりません。panel_node.pyは起動していますか?")
            return False
        # 標準のトリガーで、サーバーに要求する。
        fut = self.shuffle_cli.call_async(Trigger.Request())
        t0 = time.time()
        while not fut.done() and time.time() - t0 < timeout:
            time.sleep(0.05)
        return fut.done()

    # ------------------------------------------------------------------
    # 動作
    # ------------------------------------------------------------------
    def go(self, tip_x, tip_y, tip_z, cartesian):
        """アームの指先をロボット座標(tip_x, tip_y, tip_z)に移動させる
           moveit2の動作対象は手の原点。
           戻り値: 成功時True、失敗時False
        """
        hand_x = tip_x - self.tcp_offset # 手の原点
        # cartesian==Trueなら真っ直ぐ。o.w. 自由な軌道
        self.moveit2.move_to_pose(position=[hand_x, tip_y, tip_z],
                                  quat_xyzw=QUAT_FORWARD, cartesian=cartesian)
        ok = self.moveit2.wait_until_executed()
        if ok is False:
            self.get_logger().warn(f"move failed: tip=({tip_x:.3f},{tip_y:.3f},{tip_z:.3f}) cartesian={cartesian}")
            return False
        return True

    def read_tip(self):
        """指先のロボット座標(x, y, z)を順運動学で取得しタプルで返す。
           失敗したらNoneを返す。
        """
        try:
            fk = self.moveit2.compute_fk()
            if isinstance(fk, list):
                fk = fk[0]
            q = fk.pose.position # 手の原点のロボット座標
            return (q.x + self.tcp_offset, q.y, q.z) # 手先のロボット座標
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"FK failed: {e}")
            return None

    def touch(self, x, y, z):
        """ロボット座標(x, y, z)にタッチする。
           手前 → 真っ直ぐ前進（タッチ） → 真っ直ぐ後退。
           戻り値: 実際にタッチした指先のロボット座標（順運動学）。失敗の場合Noneを返す。
        """
        if not self.go(x - self.pre_touch, y, z, cartesian=False):
            return None
        if not self.go(x, y, z, cartesian=True): # 真っ直ぐx軸正方向
            return None
        time.sleep(0.2)
        tip = self.read_tip() # 指先のロボット座標
        if tip is None:
            self.get_logger().warn("指先位置を取れなかったため、指令位置を使います(採点は甘くなります)")
            tip = (x, y, z)
        self.go(x - self.pre_touch, y, z, cartesian=True) # 真っ直ぐx軸負方向
        return tip

    def go_ready(self):
        """ロボットを待機姿勢にする。"""
        self.moveit2.move_to_configuration(READY)
        self.moveit2.wait_until_executed()

    # ------------------------------------------------------------------
    # RViz表示
    # ------------------------------------------------------------------
    def _marker(self, mid, mtype, action=Marker.ADD):
        """RViz表示用マーカーのインスタンスを生成し返す。"""
        m = Marker()
        m.header.frame_id = self.frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "task" # 名前空間
        m.id = mid
        m.type = mtype
        m.action = action
        m.pose.orientation.w = 1.0 # 回転なし
        return m

    def cell_center(self, idx):
        """マス番号idx（行優先順）のマスの中心位置のロボット座標を返す"""
        row, col = divmod(idx, 3)
        pitch = self.panel_size / 3.0
        return (self.panel_x, self.panel_cy + (1 - col) * pitch, self.panel_cz + (1 - row) * pitch)

    def mark_cell(self, idx, ok):
        """マス番号idxのマス（タッチしたマス）上に表示するマーカーをトピックで配信する。
           正解の場合、緑の不透明な板。不正解の場合、赤い不透明の板。
           ok: 正解の場合True。不正解の場合False。
        """
        x, y, z = self.cell_center(idx) # マス番号idxのマスの中心位置のロボット座標
        pitch = self.panel_size / 3.0 # マス1辺の長さ(m)

        # タッチした位置に対応するマーカーの生成
        m = self._marker(400 + idx, Marker.CUBE)
        # x座標は、パネル板より4 mm手前。
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x - 0.004, y, z
        # マーカーの厚み4 mm、幅および高さはマスの92%
        m.scale.x, m.scale.y, m.scale.z = 0.004, pitch * 0.92, pitch * 0.92
        # マーカーの色は、正解の場合半透明の緑、不正解の場合半透明の赤
        m.color.r, m.color.g, m.color.b, m.color.a = (0.1, 0.8, 0.2, 0.45) if ok else (0.9, 0.1, 0.1, 0.55)
        self.marker_pub.publish(MarkerArray(markers=[m]))

    def clear_marks(self):
        """タッチしたマスが正解だったか否かを示すマーカーをRVizから削除するトピックを配信する。"""
        # https://docs.ros2.org/foxy/api/visualization_msgs/msg/MarkerArray.html
        arr = MarkerArray()
        for i in range(9):
            arr.markers.append(self._marker(400 + i, Marker.CUBE, Marker.DELETE))
        self.marker_pub.publish(arr)

    def clear_status(self):
        """以前の進行表示を消去するマーカーをトピックで配信する。"""
        arr = MarkerArray()
        for mid in range(300, 307):
            arr.markers.append(self._marker(mid, Marker.TEXT_VIEW_FACING, Marker.DELETE))
        self.marker_pub.publish(arr)

    def set_status(self, text, bg=(0.08, 0.08, 0.10)):
        """
        進行表示のマーカー（白文字、太字ふう）をトピックに配信する。
        文字は、指定したロボット座標を文字全体の中心として描かれるので、行数に応じて
        中心を上へずらし、パネルの上端に重ならないようにする。
        RVizの空白文字は広すぎるので使わない。
        """
        lines = ([f"[{self.label}]"] if self.label else []) + text.split("\n")
        text = "\n".join(lines)
        size = self.status_size # 文字の高さ
        block_h = len(lines) * size * 1.25   # 文字全体の高さの目安（少し大きめ）
        x = self.panel_x - 0.02
        y = self.panel_cy
        z_top_of_panel = self.panel_cz + self.panel_size / 2.0 + 0.02
        z = z_top_of_panel + 0.04 + block_h / 2.0 + self.status_dz # 文字全体の中心のz座標

        arr = MarkerArray()
        if self.status_bg:  # 進行表示の背景に暗い板を置く場合
            n = max(len(s) for s in lines)
            b = self._marker(301, Marker.CUBE)
            b.pose.position.x, b.pose.position.y, b.pose.position.z = x + 0.006, y, z
            b.scale.x = 0.004 # 厚み
            b.scale.y = size * 0.62 * n + 0.04 # 幅
            b.scale.z = block_h + 0.02 # 高さ
            b.color.r, b.color.g, b.color.b, b.color.a = bg[0], bg[1], bg[2], 0.9
            arr.markers.append(b)

        # 太字ふう: 斜め4方向に1.5 mm ずらした文字を重ね、最後に中心の文字を描く
        d = 0.0015
        offsets = ([(d, d), (-d, d), (d, -d), (-d, -d)] if self.status_bold else []) + [(0.0, 0.0)]
        for k, (dy, dz) in enumerate(offsets):
            m = self._marker(302 + k, Marker.TEXT_VIEW_FACING)
            # 文字全体の真ん中のロボット座標
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y + dy, z + dz
            m.scale.z = size # 文字の高さ
            m.color.r = m.color.g = m.color.b = 1.0 # 白
            m.color.a = 1.0 # 不透明
            m.text = text
            arr.markers.append(m)

        self.marker_pub.publish(arr)
        self.get_logger().info(text.replace("\n", " | "))

    # ------------------------------------------------------------------
    # タスクの実行
    # ------------------------------------------------------------------
    def run_trial(self, trial):
        """trial回目のタスクを実行し、その結果を返す。"""
        self.clear_marks() # 前回タッチしたマスが正解だったか否かを示すマーカーを削除する
        self.set_status(f"trial:{trial}/{self.trials}\nshuffle") # 進行表示
        count = self.msg_count # トピックの受信回数
        if not self.shuffle(): # 数字を並べ替える。
            return None # 数字の並べ替えサービスのサーバがない。
        # 何度か認識結果が一致するまで待つ。
        got = self.wait_stable(count, self.stable_frames)
        if got is None:
            self.get_logger().error("認識結果が安定しません。perception_node.pyは動いていますか?")
            return None
        # 認識した数字列、行優先順の各マス中心のロボット座標、認識結果が揃ったか否か
        digits, points, stable = got
        if not stable:
            self.get_logger().warn("認識結果が揃いませんでした。最新の結果で続けます")
        truth = list(self.truth) if self.truth else None # 正解の数字列のリスト
        perception_ok = (truth == digits) if truth else None
        self.get_logger().info(f"認識結果: {digits[0:3]} / {digits[3:6]} / {digits[6:9]}"
                               f"  (認識は{'正しい' if perception_ok else '誤り' if perception_ok is False else '不明'})")

        # 認識した数字列の小さい順に各マスにタッチする。
        order = plan_order(digits) # タッチする順序に並べたマス番号のリスト
        correct = 0
        t0 = time.time()
        for step, cell in enumerate(order):
            expected = step + 1 # タッチする数字
            self.set_status(f"trial:{trial}/{self.trials}\nnext:{expected}|OK:{correct}/9")
            x, y, z = points[cell] # タッチ先のロボット座標
            tip = self.touch(x, y, z) # tipは、実際にタッチしたロボット座標
            if tip is None:
                self.get_logger().warn(f"  step {expected}: 動作に失敗")
                break # 実行中止

            touched = None
            if abs(tip[0] - self.panel_x) <= self.touch_tol:
                # タッチしたマス番号（行優先）
                touched = tip_to_cell(tip[1], tip[2], self.panel_cy, self.panel_cz, self.panel_size)
            # タッチしたマスの正解の数字
            touched_digit = truth[touched] if (touched is not None and truth) else None
            ok = (touched_digit == expected) # 正解ならTrue。不正解ならFalse。
            if touched is not None:
                self.mark_cell(touched, ok) # 正解、不正解のRViz表示を配信する
            if ok:
                correct += 1
            self.get_logger().info(
                f"  step {expected}: タッチしたマス={touched} (正解の数字={touched_digit}) -> {'OK' if ok else 'MISS'}")
            if not ok and self.abort_on_miss: # 失敗
                break

        elapsed = time.time() - t0
        success = (correct == 9) # 全てのマスが正解ならTrue。さもなければFalse。
        self.set_status(f"{'SUCCESS' if success else 'FAILED'}\n{correct}/9|{elapsed:.1f}s",
                        bg=(0.05, 0.45, 0.15) if success else (0.65, 0.08, 0.08))
        time.sleep(0.8)
        self.go_ready() # ロボットを待機姿勢にする
        # trial回目の実行結果を返す。
        return dict(trial=trial, success=int(success), correct=correct,
                    time_sec=round(elapsed, 1),
                    perception_ok="" if perception_ok is None else int(perception_ok),
                    stable=int(stable))

    # ------------------------------------------------------------------
    def run(self):
        self.clear_status() # 以前の進行表示を消す
        self.get_logger().info("待機姿勢へ移動")
        self.go_ready() # ロボットを待機姿勢にする

        t0 = time.time()
        # トピックに正解が配信されるのを待つ。
        while self.truth is None and time.time() - t0 < 5.0:
            time.sleep(0.1)
        if self.truth is None:
            self.get_logger().warn("/panel/truthが届いていません。採点できません (panel_node.pyを確認)")

        # self.trials回タスクを実行する(self.run_trials)。
        results = []
        for trial in range(1, self.trials + 1):
            r = self.run_trial(trial)
            if r is None:
                self.get_logger().error(f"trial {trial}を中止しました")
                break
            results.append(r)

        # 結果の表示およびファイルへの追記
        self.summarize(results)

    def summarize(self, results):
        if not results:
            return
        n = len(results)
        succ = sum(r["success"] for r in results)
        mean_t = sum(r["time_sec"] for r in results) / n
        known = [r["perception_ok"] for r in results if r["perception_ok"] != ""]
        perc = f", 認識が全マス正しかった試行 {sum(known)}/{len(known)}" if known else ""
        self.get_logger().info(f"===== {n} 試行: 成功 {succ}/{n} ({100 * succ / n:.0f}%), "
                               f"平均所要時間 {mean_t:.1f}s{perc} =====")
        # 結果をファイルself.csv_pathに追記する。
        new_file = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["trial", "success", "correct", "time_sec", "perception_ok", "stable"])
            if new_file:
                w.writeheader()
            w.writerows(results)
        self.get_logger().info(f"結果を{self.csv_path}に追記しました")


def main():
    rclpy.init()
    node = TaskNode()
    # TaskNodeのマルチスレッド実行設定
    executor = MultiThreadedExecutor(4)
    executor.add_node(node) # マルチスレッド実行対象の登録（コールバック関数）
    # バックグラウンドでメッセージの受信監視用スレッドを起動する。
    #   メッセージの受信時に対応するコールバック関数が実行される。
    #   daemon=Trueによって、主スレッドが終了するときこれらのスレッドも終了する。
    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    # スレッドの起動待ち。
    node.create_rate(1.0).sleep()

    try:
        node.run() # 主スレッドでTaskNodeの関数runを実行する。
    finally:
        rclpy.shutdown()
        spin_thread.join()


if __name__ == "__main__":
    main()
