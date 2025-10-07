import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO  # YOLOv8 Python版

# 学習済みYOLOモデルの読み込み（パスは適宜変更）
yolo_model = YOLO('../model/knife.pt')

# MediaPipe Hands初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# カメラ映像のキャプチャを開始
cap = cv2.VideoCapture(0)

# メインループ：カメラ映像を処理
while cap.isOpened():
    ret, frame = cap.read()  # フレームを取得
    if not ret:  # フレームが取得できなければ終了
        break

    # フレームをRGBに変換（MediaPipeはRGB画像を使用）
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = frame.shape  # フレームの高さと幅を取得

    # MediaPipe Handsで手を検出
    results = hands.process(img_rgb)

    # YOLOで包丁を検出
    yolo_results = yolo_model(frame)[0]  # 推論結果の1つ目を取得

    # 包丁クラスのIDは環境により異なるので要調整（例: クラスID 0が包丁）
    knife_boxes_scores = []  # 包丁のバウンディングボックスとスコアを格納するリスト
    for box, cls, score in zip(yolo_results.boxes.xyxy, yolo_results.boxes.cls, yolo_results.boxes.conf):
        if int(cls) == 0:  # クラスIDが0の物体を包丁と仮定
            xmin, ymin, xmax, ymax = map(int, box)  # バウンディングボックスの座標を整数に変換
            knife_boxes_scores.append((score, [xmin, ymin, xmax, ymax]))

    # スコアが最も高い包丁ボックスを1つ選択
    if knife_boxes_scores:
        knife_boxes_scores.sort(key=lambda x: x[0], reverse=True)  # スコアでソート
        knife_boxes = [knife_boxes_scores[0][1]]  # スコアが最大の箱のみ選択
    else:
        knife_boxes = []  # 包丁が検出されなかった場合は空リスト

    # 手の中心座標を計算する関数
    def get_hand_center(landmarks):
        xs = [lm.x for lm in landmarks.landmark]  # x座標のリスト
        ys = [lm.y for lm in landmarks.landmark]  # y座標のリスト
        cx = int(np.mean(xs) * img_w)  # x座標の平均を計算して画像幅を掛ける
        cy = int(np.mean(ys) * img_h)  # y座標の平均を計算して画像高さを掛ける
        return cx, cy

    # 検出された手の中心座標リスト
    hand_centers = []
    if results.multi_hand_landmarks:  # 手が検出された場合
        for hand_landmarks in results.multi_hand_landmarks:
            cx, cy = get_hand_center(hand_landmarks)  # 手の中心座標を計算
            hand_centers.append((cx, cy))  # リストに追加
            # 手のランドマークを描画
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 各手の中心に最も近い包丁バウンディングボックスを描画
    for cx, cy in hand_centers:
        min_dist = float('inf')  # 最小距離を無限大で初期化
        nearest_box = None  # 最も近いバウンディングボックス
        for box in knife_boxes:
            xmin, ymin, xmax, ymax = box  # バウンディングボックスの4点の座標を取得
            bx = (xmin + xmax) // 2  # バウンディングボックスの中心x座標
            by = (ymin + ymax) // 2  # バウンディングボックスの中心y座標
            dist = np.hypot(cx - bx, cy - by)  # ユークリッド距離を計算
            if dist < min_dist:  # より近い場合は更新
                min_dist = dist
                nearest_box = box
        if nearest_box:  # 最も近いバウンディングボックスが存在する場合
            xmin, ymin, xmax, ymax = nearest_box
            # バウンディングボックスを描画
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            # ラベルを描画
            cv2.putText(frame, 'Knife', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 処理結果を表示
    cv2.imshow("MediaPipe Hands + YOLO Knife Detection", frame)

    # 「q」キーが押されたら終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放
cap.release()
cv2.destroyAllWindows()

# 詳細なコード説明
# このスクリプトは、リアルタイムでカメラ映像を処理し、手の検出と包丁の検出を行います。以下に、各部分の詳細な説明を記載します。

# 初期化部分
# YOLOモデルの初期化
# - YOLO('../model/knife.pt')
#   - YOLOv8モデルを初期化します。
#   - '../model/knife.pt' は学習済みモデルのパスです。
#   - このモデルは、包丁を検出するために事前にトレーニングされています。
#   - 必要に応じて、モデルのパスやクラスIDを変更してください。

# MediaPipe Handsの初期化
# - mp.solutions.hands
#   - MediaPipeの手検出モジュールを使用します。
# - Handsクラスの引数：
#   - static_image_mode=False
#     - 動画ストリームのような連続フレームで使用する場合はFalseに設定します。
#   - max_num_hands=2
#     - 検出する手の最大数を指定します。
#     - この場合、最大2つの手を検出します。
#   - min_detection_confidence=0.7
#     - 手を検出するための信頼度の閾値を設定します。
#     - 値が高いほど、検出の精度が高くなりますが、検出されにくくなります。

# ランドマークのプロパティ
# - landmarks.landmark
#   - 各ランドマークのリストを保持します。
#   - 各ランドマークには以下のプロパティがあります：
#     - x：画像幅に対する相対的なx座標（0.0〜1.0）
#     - y：画像高さに対する相対的なy座標（0.0〜1.0）
#     - z：カメラからの相対的な深度情報（負の値はカメラに近いことを示します）。

# YOLO推論結果
# - yolo_results.boxes
#   - 検出されたオブジェクトのバウンディングボックス情報を保持します。
#   - xyxy：バウンディングボックスの座標（左上と右下のx, y座標）。
#   - cls：検出されたオブジェクトのクラスID。
#   - conf：検出の信頼度スコア。