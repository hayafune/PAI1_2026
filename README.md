# 数字の順番タッチ — AI×ロボティクスによる創造的タスク

チンパンジーのアイちゃん（京都大学霊長類研究所）の数字課題（パネル上に散らばった数字1〜9を小さい順にタッチする）を、
ロボットアーム(Franka Panda)のシミュレーションで再現した。

- **AIの役割**: パネル画像から数字を読み取るCNN。
- **工夫**: 「数字1〜9が1回ずつ現れる」という制約条件を利用して、CNNの出力を全体で最適に並べ直す（制約条件を利用した復号）。
- **採点**: ロボットの指先の実際の位置（順運動学）からタッチしたマスを求め、正解の配置と照合する。認識には正解を使わない。

## 用語

<!--動画・レポート・このREADMEで、次の呼び方にそろえている。-->

| 用語 | 意味 | 動画の画面のラベル / コードの指定 |
|---|---|---|
| きれいな画像 | ノイズ・ぼかし・位置ずれなしの画像（`panel_node.py`の既定値） | `clean` |
| 劣化した画像 | ノイズ（標準偏差40）、ぼかし（カーネル9）、数字の位置ずれ（14ピクセル）を加えた画像 | — |
| 制約条件を利用した | 「数字1〜9が1回ずつ」という制約条件を使い、認識結果を全体で最適に並べ直す | `degraded-constrained` / `perception_node.py`（既定値） |
| 制約条件を利用せず | マスごとに独立に確率が最大の数字を選ぶ | `degraded-unconstrained` / `perception_node.py -p unique_decode:=false` |
| 3フレームの合意 | 全てのマスの数字の認識結果が3回連続で一致するまで待ってから、ロボットによるタスクを開始する方式 | `task_node.py -p stable_frames:=3` |

動画およびレポートでは、次のタスク名を使用する。

| タスク | 画像条件 | 認識方式 |
|---|---|---|
| タスクA | きれいな画像 | 制約条件を利用した |
| タスクB | 劣化した画像 | 制約条件を利用せず |
| タスクC | 劣化した画像 | 制約条件を利用した |
| タスクD | 劣化した画像 | 制約条件を利用した+3フレームの合意（動画に含めていない） |

## 動作環境

- Windows 11 + WSL2(Ubuntu 24.04) / ROS 2 Jazzy / MoveIt 2(Fake Hardware) / pymoveit2
- ROS側Python: システムのPython 3.12(OpenCV 4.6, NumPy 1.26)。TensorFlowは不要
- 学習用: TensorFlow CPU 2.21, `opencv-python-headless<5`（別の仮想環境`~/venv_tf`）
- GPU・実機は使用しない

**限界**: Fake Hardwareは運動学だけで、物理接触やカメラセンサのモデルをもたない。
カメラ画像は、`panel_render.py`が描いたパネル画像（仮想カメラの画像と呼ぶ）であり、「タッチした」は、指先がパネル面に到達したことを意味する。

## ファイル

| ファイル | 役割 |
|---|---|
| `panel_render.py` | パネル画像の描画、ピクセル座標↔パネル座標(m)の変換、9マスの切り出し（ROS非依存）。 |
| `panel_node.py` | 数字1〜9をランダムに配置する。仮想カメラの画像・RVizによる表示用のマーカー（パネルの板、各マスの数字、色などの情報）・正解を配信する。サービス通信`/panel/shuffle`で数字の並べ替えを行う。 |
| `train_cnn.py` | プログラムで生成した画像でCNNを学習し、重みを`digit_cnn.npz`に書き出す。Python仮想環境venv_tfで実行する。 |
| `digit_cnn_np.py` | NumPyだけで作成したCNN推論コード。ROS側で使用する。 |
| `perception_node.py` | 画像から数字を認識し位置とともに配信する。制約条件を利用した復号を含む。 |
| `task_node.py` | 認識した数字の小さい順にタッチし、採点してファイル`results*.csv`に記録する。 |
| `image_viewer.py` | 仮想カメラの画像を表示する。RVizのImageディスプレイがWSL2で落ちるので、別途作成した。 |
| `touch_one_point.py` | 各マスに指先が届くことの確認用。指定した1マス（`-p cell:=マス番号`。行優先で左上が0）、または9マス全部（`-p mode:=grid`）をタッチする。 |
| `eval_offline.py` | ROSなしで画像条件ごとの認識精度を診断する。 |
| `digit_cnn.npz` | 学習済みの重み。 |
| `results_*.csv` | 実験の記録。 |

## 実行手順

### 1. 学習（初回または再学習するとき）

```bash
sudo apt install -y python3-pip python3-venv
python3 -m venv ~/venv_tf
source ~/venv_tf/bin/activate
pip install --upgrade pip
pip install tensorflow-cpu "opencv-python-headless<5"
unset PYTHONPATH
python train_cnn.py            # 重みファイルdigit_cnn.npzが生成される（CPUで数分程度）
deactivate
```

**重要**: 学習側とROS側との間でOpenCVの版が異なると、`cv2.putText`による描画が変わり認識精度が大きく落ちる
（5.0で学習 → 4.6で推論: 100% → 約52%）。学習はROS側と同じ4系で行うこと。

### 2. 診断（任意）

```bash
# 条件ごとにパネル画像を100枚生成し、認識精度を測定する。
python3 eval_offline.py --panels 100
```

### 3. 実行（別々のターミナルで）

```bash
# 1: MoveIt 2 + RViz
ros2 launch moveit_resources_panda_moveit_config demo.launch.py

# 2: パネル（RVizのMarkerArrayにTopic=/touch_markersを追加しておく）
python3 panel_node.py

# 3: 認識
python3 perception_node.py

# 4: 表示（任意）
python3 image_viewer.py

# 5: タスクA
python3 task_node.py --ros-args -p trials:=10 -p stable_frames:=1 -p csv:=results_clean.csv
```

小数のパラメタは小数点付きで指定する（`-p noise_std:=40.0`。`40`だと型エラーになる）。

**RVizの視点**（動画と同じ画角にするとき）:

1. 上のメニューのPanels→Viewsを開く（画面の右か下にパネルが出る）。
2. TypeがOrbit(rviz_default_plugins)になっていることを確認する。
3. 次の値を設定する。
   - Distance: 1.44155（カメラと中心点との距離。大きくするとカメラが遠ざかる）
   - Focal Point (X, Y, Z): (0.32197, -0.10981, 0.44531)（画面の中心にする点）
   - Yaw / Pitch: 1.9504 / -0.0452026（回り込む角度と、見下ろす角度）

### 4. 劣化した画像での比較実験

```bash
python3 panel_node.py --ros-args -p seed:=0 -p noise_std:=40.0 -p blur_ksize:=9 -p jitter_px:=14.0

# タスクB: 制約条件を利用せず
python3 perception_node.py --ros-args -p unique_decode:=false
python3 task_node.py --ros-args -p trials:=20 -p stable_frames:=1 -p csv:=results_severe_raw.csv

# タスクC: 制約条件を利用した
python3 perception_node.py
python3 task_node.py --ros-args -p trials:=20 -p stable_frames:=1 -p csv:=results_severe_decoded.csv

# タスクD: 制約条件を利用した+3フレームの合意
python3 task_node.py --ros-args -p trials:=20 -p stable_frames:=3 -p csv:=results_severe_decoded_stable.csv
```

## 結果

| タスク | 画像条件 | 認識方式 | 成功 | 成功率 | 成功した試行の平均実行時間 |
|---|---|---|---|---|---|
| タスクA | きれいな画像 | 制約条件を利用した | 10/10 | 100% | 29.0秒 |
| タスクB | 劣化した画像(noise 40, blur 9, jitter 14) | 制約条件を利用せず | 8/20 | 40% | 29.4秒 |
| タスクC | 劣化した画像 | 制約条件を利用した | 19/20 | 95% | 28.9秒 |
| タスクD | 劣化した画像 | 制約条件を利用した+3フレームの合意 | 20/20 | 100% | 28.5秒 |

- 制約条件を利用しない場合と利用した場合との成功率の差は、フィッシャーの正確確率検定でp=0.0004であった。
- 制約条件を利用した場合(19/20)と、さらに3フレームの合意を加えた場合(20/20)との差は、試行数が少ないので有意とはいえない。
- ROSなしの認識精度診断スクリプト（`eval_offline.py`）が予測したパネル全体の正解率（制約条件を利用せず: 42%、制約条件を利用した: 91%）は、ロボットでの成功率（40%、95%）とよく一致した。
- CSVの列`stable`は、認識結果が3フレーム連続で一致するまで待機する処理が成立したか否かを示す（成立=1、タイムアウト=0）。
  この待機を使わない実験(`stable_frames:=1`)では常に1になる。タスクB・C（`results_severe_raw.csv`、`results_severe_decoded.csv`）は待機を使っていないので、`stable`は意味をもたない。
  タスクD（`results_severe_decoded_stable.csv`）は、20試行全てで1であった。
- `results_clean.csv`（きれいな画像）は、この列を追加する前に取得したので、`stable`列がない。成功率および所要時間の計算には影響しない。
- 試行数は各20回と少ないので、成功率の95%信頼区間（真の成功率と考えられる範囲）は広い（制約条件を利用せず: 22〜61%、制約条件を利用した: 76〜99%）。

## 既知の限界

- 運動学だけのシミュレーションであり、物理接触、カメラセンサ、照明などは扱っていない。
- 数字は3×3の固定位置に並ぶことを前提としている。ランダムな位置への拡張（物体検出）は未実装。
- 画像と正解とを別のトピックで配信しているので、並べ替えの直後は、受信側で1フレームの食い違いが起こりうる。受信側で並べ替え直後の結果を除外して対処したが、1つのメッセージにまとめたほうが確実であった。
- CNNは位置ずれ12ピクセルまでで学習している。14〜16ピクセルでは精度が落ちる。
- 画像を劣化させる処理の種類および程度は、制約条件を利用する効果が見えやすいものを選択した。したがって、本課題における40%および95%という成功率の差は、他の種類または程度にそのまま当てはまるとは限らない。
- 制約条件を利用しない場合、認識が誤っているにもかかわらず、同じ数字が2か所に出たときの並べ替えの順序がたまたま正解と一致して成功する試行があった。成功率は、認識の正しさとは必ずしも一致しない。
- 各条件20回と試行数が少ないので、差の大きさの推定は粗い。
