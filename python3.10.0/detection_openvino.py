import gc
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# カメラ設定
input_length = 640  # 例：モデルによる入力画像サイズ
model_path = "C:/Users/mi241326/idea/procon/createModel/python3.10.0/model/model1/best_openvino_model"
device = "intel:gpu" # OpenVINO GPUデバイス
task = "detect" # 検出タスク
detection_interval = 300  # 推論間隔（フレーム数）
confidence_threshold = 0.50  # 包丁検出の信頼度閾値

# YOLOモデルのロード
yolo_model = YOLO(model_path, task=task)

# MediaPipe Handsの初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# カメラ設定
cap = cv2.VideoCapture(0)
# カメラ設定はループ外で一度だけ
cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_length)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_length)

# ループ外で関数定義
def get_hand_center(landmarks, width, height):
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    cx = int(np.mean(xs) * width)
    cy = int(np.mean(ys) * height)
    return cx, cy

# フレームループ
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # RGB変換とサイズ取得
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = frame.shape

    # MediaPipe Handsで手検出
    results = None
    try:
        results = hands.process(img_rgb)
    except Exception as e:
        print("MediaPipe Handsエラー:", e)
        results = None

    # YOLO推論（間引き）
    yolo_results = None
    try:
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % detection_interval == 0:  # 5フレームに一度だけ
            yolo_results = yolo_model(frame, device=device, conf=confidence_threshold)[0]
    except Exception as e:
        print("YOLO推論エラー:", e)
        yolo_results = None

    # 推論結果の処理
    knife_boxes_scores = []

    if yolo_results is not None:
        # Resultsオブジェクトか配列か判定
        if hasattr(yolo_results, 'boxes'):
            for box, cls, score in zip(yolo_results.boxes.xyxy, yolo_results.boxes.cls, yolo_results.boxes.conf):
                if int(cls) == 0:  # 包丁のクラスID
                    xmin, ymin, xmax, ymax = map(int, box)
                    knife_boxes_scores.append((score, [xmin, ymin, xmax, ymax]))
        elif isinstance(yolo_results, np.ndarray):
            for det in yolo_results:
                xmin, ymin, xmax, ymax, conf, cls_id = det
                if int(cls_id) == 0:  # 包丁のクラスID
                    xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
                    knife_boxes_scores.append((conf, [xmin, ymin, xmax, ymax]))
        else:
            print("未知のYOLO結果形式")
            # 例外発生もしくはスキップ
    # スコア最大の包丁ボックス
    if knife_boxes_scores:
        knife_boxes_scores.sort(key=lambda x: x[0], reverse=True)
        highest_score, knife_boxes = knife_boxes_scores[0] 
    else:
        highest_score, knife_boxes = None, None

    # 手の中心座標
    hand_centers = []
    if results is not None and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            cx, cy = get_hand_center(hand_landmarks, img_w, img_h)
            hand_centers.append((cx, cy))
            # 手のランドマーク描画
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 包丁と手の座標を利用した描画
    for cx, cy in hand_centers:
        min_dist = float('inf')
        nearest_box = None
        nearest_score = None
        for score, box in knife_boxes_scores:
            xmin, ymin, xmax, ymax = box
            bx = (xmin + xmax) // 2
            by = (ymin + ymax) // 2
            dist = np.hypot(cx - bx, cy - by)
            if dist < min_dist:
                min_dist = dist
                nearest_box = box
                nearest_score = score
        if nearest_box:
            xmin, ymin, xmax, ymax = nearest_box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            score_text = f'Knife: {nearest_score:.2f}'
            cv2.putText(frame, score_text, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # メモリ解放
    if yolo_results is not None:
        del yolo_results
    if results is not None:
        del results
    gc.collect()

    # 画像表示
    cv2.imshow("MediaPipe Hands + YOLO Knife Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 各種リソースの解放
cap.release()
cv2.destroyAllWindows()
hands.close()
