import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO  # YOLOv8 Python版

# 学習済みYOLOモデルの読み込み（パスは適宜変更）
yolo_model = YOLO('../model/knife.pt')

# MediaPipe Hands初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)  # カメラ映像

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = frame.shape

    # MediaPipe Handsで手検出
    results = hands.process(img_rgb)

    # YOLOで包丁推論
    yolo_results = yolo_model(frame)[0]  # 推論結果の1つ目
    
    # 包丁クラスのIDは環境により異なるので要調整、例としてclass 0が包丁
    # knife_boxes = []
    # for box, cls in zip(yolo_results.boxes.xyxy, yolo_results.boxes.cls):
    #     if int(cls) == 0:  # クラスIDが0の物体を包丁と仮定
    #         xmin, ymin, xmax, ymax = map(int, box)
    #         knife_boxes.append([xmin, ymin, xmax, ymax])

    # 包丁ボックスとスコアのペアを作る
    knife_boxes_scores = []
    for box, cls, score in zip(yolo_results.boxes.xyxy, yolo_results.boxes.cls, yolo_results.boxes.conf):
        if int(cls) == 0:  # クラスIDが0の物体を包丁と仮定
            xmin, ymin, xmax, ymax = map(int, box)
            knife_boxes_scores.append((score, [xmin, ymin, xmax, ymax]))

    # スコアが最も高い包丁ボックスを1つ選択
    if knife_boxes_scores:
        knife_boxes_scores.sort(key=lambda x: x[0], reverse=True)
        knife_boxes = [knife_boxes_scores[0][1]]  # スコアが最大の箱のみ
    else:
        knife_boxes = []

    # 手の中心取得関数
    def get_hand_center(landmarks):
        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]
        cx = int(np.mean(xs) * img_w)
        cy = int(np.mean(ys) * img_h)
        return cx, cy

    # 手の中心座標リスト
    hand_centers = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            cx, cy = get_hand_center(hand_landmarks)
            hand_centers.append((cx, cy))
            # 手のランドマークを描画
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 各手の中心に最も近い包丁バウンディングボックスを描画
    for cx, cy in hand_centers:
        min_dist = float('inf')
        nearest_box = None
        for box in knife_boxes:
            xmin, ymin, xmax, ymax = box
            bx = (xmin + xmax) // 2
            by = (ymin + ymax) // 2
            dist = np.hypot(cx - bx, cy - by)
            if dist < min_dist:
                min_dist = dist
                nearest_box = box
        if nearest_box:
            xmin, ymin, xmax, ymax = nearest_box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, 'Knife', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("MediaPipe Hands + YOLO Knife Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
