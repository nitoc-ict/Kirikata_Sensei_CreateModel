import gc
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

input_length = 640
model_path = "C:/Users/mi241326/idea/procon/createModel/python3.10.0/model/model1/best_openvino_model"
device = "intel:gpu"
task = "detect"
detection_interval = 1
confidence_threshold = 0.50

yolo_model = YOLO(model_path, task=task)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_length)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_length)


def get_hand_center(landmarks, width, height):
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    cx = int(np.mean(xs) * width)
    cy = int(np.mean(ys) * height)
    return cx, cy


def calc_angle(p1, p2, p3):
    # 3点p1,p2,p3の角度を計算（p2が頂点）
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)


def is_finger_extended(hand_landmarks, finger_indices):
    # finger_indicesは指の3点インデックス（例：親指 (1,2,3), 人差し指 (5,6,7)など）
    angles = []
    for i in range(len(finger_indices) - 2):
        p1 = hand_landmarks.landmark[finger_indices[i]]
        p2 = hand_landmarks.landmark[finger_indices[i + 1]]
        p3 = hand_landmarks.landmark[finger_indices[i + 2]]
        angle = calc_angle(p1, p2, p3)
        angles.append(angle)
    # 角度がある閾値以上なら伸びている判定（例: 160°以上）
    return all(a > 160 for a in angles)


# 指の関節インデックス（MediaPipe Hands）
finger_points = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20]
}

prev_knife_area = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = frame.shape

    results = None
    try:
        results = hands.process(img_rgb)
    except Exception as e:
        print("MediaPipe Handsエラー:", e)
        results = None

    yolo_results = None
    try:
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % detection_interval == 0:
            yolo_results = yolo_model(frame, device=device, conf=confidence_threshold)[0]
    except Exception as e:
        print("YOLO推論エラー:", e)
        yolo_results = None

    knife_boxes_scores = []
    if yolo_results is not None:
        if hasattr(yolo_results, 'boxes'):
            for box, cls, score in zip(yolo_results.boxes.xyxy, yolo_results.boxes.cls, yolo_results.boxes.conf):
                if int(cls) == 0:
                    xmin, ymin, xmax, ymax = map(int, box)
                    box_area = (xmax - xmin) * (ymax - ymin)
                    knife_boxes_scores.append((score, [xmin, ymin, xmax, ymax], box_area))
        elif isinstance(yolo_results, np.ndarray):
            for det in yolo_results:
                xmin, ymin, xmax, ymax, conf, cls_id = det
                if int(cls_id) == 0:
                    xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
                    box_area = (xmax - xmin) * (ymax - ymin)
                    knife_boxes_scores.append((conf, [xmin, ymin, xmax, ymax], box_area))

    if knife_boxes_scores:
        knife_boxes_scores.sort(key=lambda x: x[0], reverse=True)
        highest_score, knife_boxes, knife_area = knife_boxes_scores[0]
    else:
        highest_score, knife_boxes, knife_area = None, None, None

    hand_centers = []
    extended_finger_counts = []
    if results is not None and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            cx, cy = get_hand_center(hand_landmarks, img_w, img_h)
            hand_centers.append((cx, cy))
            # 指の伸びている本数カウント（人差し指・中指・薬指あたり）
            count_extended = 0
            for finger in ["index", "middle", "ring"]:
                if is_finger_extended(hand_landmarks, finger_points[finger]):
                    count_extended += 1
            extended_finger_counts.append(count_extended)
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 危険判定ロジック
    danger_texts = []
    for idx, (cx, cy) in enumerate(hand_centers):
        if knife_boxes is None:
            break
        xmin, ymin, xmax, ymax = knife_boxes
        bx = (xmin + xmax) // 2
        by = (ymin + ymax) // 2
        dist = np.hypot(cx - bx, cy - by)
        # 包丁の面積変化で近づき推定（前面積があれば変化率）
        area_ratio = None
        if prev_knife_area is not None and knife_area is not None:
            area_ratio = knife_area / prev_knife_area
        else:
            area_ratio = 1.0

        # 指の伸び本数
        ext_count = extended_finger_counts[idx]

        # 距離・近づきと指の伸びに応じて危険判定
        danger = False
        if dist < 150 and ext_count >= 2 and area_ratio >= 1.0:
            danger = True

        if danger:
            text = "Danger: fingers extended"
            color = (0, 0, 255)
        else:
            text = "Safe"
            color = (0, 255, 0)

        danger_texts.append((text, color))

        # 包丁ボックス描画
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, f"Knife: {highest_score:.2f}", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # 手の中心にも警告文字描画
        cv2.putText(frame, text, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    prev_knife_area = knife_area

    if yolo_results is not None:
        del yolo_results
    if results is not None:
        del results
    gc.collect()

    cv2.imshow("MediaPipe Hands + YOLO Knife Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
