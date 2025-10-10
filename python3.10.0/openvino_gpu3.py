import gc
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO


# カメラ設定
input_length = 640
model_path = "C:/Users/mi241326/idea/procon/createModel/python3.10.0/model/model1/best_openvino_model"
device = "intel:gpu"
task = "detect"
detection_interval = 2
confidence_threshold = 0.50

# 危険判定のパラメータ
DANGER_DISTANCE = 50  # 包丁と手の危険距離（ピクセル）
MIN_EXTENDED_FINGERS = 3  # 危険と判定する最小伸展指数

# YOLOモデルのロード
yolo_model = YOLO(model_path, task=task)

# MediaPipe Handsの初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# カメラ設定
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_length)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_length)


def get_hand_center(landmarks, width, height):
    """手の中心座標を計算"""
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    cx = int(np.mean(xs) * width)
    cy = int(np.mean(ys) * height)
    return cx, cy


def is_finger_curled(landmarks, finger_tip_id, finger_pip_id):
    """指が曲がっているかを判定（簡易版）"""
    tip = landmarks.landmark[finger_tip_id]
    pip = landmarks.landmark[finger_pip_id]
    wrist = landmarks.landmark[0]
    
    tip_to_wrist = np.hypot(tip.x - wrist.x, tip.y - wrist.y)
    pip_to_wrist = np.hypot(pip.x - wrist.x, pip.y - wrist.y)
    
    return tip_to_wrist < pip_to_wrist * 1.1


def count_curled_fingers(hand_landmarks):
    """曲がっている指の数をカウント"""
    curled_count = 0
    
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    index_mcp = hand_landmarks.landmark[5]
    
    thumb_to_index = np.hypot(thumb_tip.x - index_mcp.x, thumb_tip.y - index_mcp.y)
    thumb_ip_to_index = np.hypot(thumb_ip.x - index_mcp.x, thumb_ip.y - index_mcp.y)
    if thumb_to_index < thumb_ip_to_index * 1.2:
        curled_count += 1
    
    if is_finger_curled(hand_landmarks, 8, 6):
        curled_count += 1
    
    if is_finger_curled(hand_landmarks, 12, 10):
        curled_count += 1
    
    if is_finger_curled(hand_landmarks, 16, 14):
        curled_count += 1
    
    if is_finger_curled(hand_landmarks, 20, 18):
        curled_count += 1
    
    extended_count = 5 - curled_count
    return extended_count


def get_knife_center(box):
    """包丁の中心座標を計算"""
    xmin, ymin, xmax, ymax = box
    return (xmin + xmax) // 2, (ymin + ymax) // 2


def calculate_distance(point1, point2):
    """2点間の距離を計算"""
    return np.hypot(point1[0] - point2[0], point1[1] - point2[1])


def draw_danger_warning(frame):
    """危険警告を画面に描画"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    warning_text = "!! DANGER !!"
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 2.5
    thickness = 5
    
    text_size = cv2.getTextSize(warning_text, font, font_scale, thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2
    
    cv2.putText(frame, warning_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 3)
    cv2.putText(frame, warning_text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)


# フレームループ
frame_count = 0
yolo_results = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # ミラー表示

    frame_count += 1

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = frame.shape

    results = None
    try:
        results = hands.process(img_rgb)
    except Exception as e:
        print("MediaPipe Handsエラー:", e)
        results = None

    try:
        if frame_count % detection_interval == 0:
            yolo_results = yolo_model(frame, device=device, conf=confidence_threshold)[0]
    except Exception as e:
        print("YOLO推論エラー:", e)

    knife_boxes_scores = []
    if yolo_results is not None and hasattr(yolo_results, 'boxes') and len(yolo_results.boxes) > 0:
        for box, cls, score in zip(yolo_results.boxes.xyxy, yolo_results.boxes.cls, yolo_results.boxes.conf):
            if int(cls) == 0:
                xmin, ymin, xmax, ymax = map(int, box)
                knife_boxes_scores.append((score.item(), [xmin, ymin, xmax, ymax]))

    # ### 変更: 手の情報を左右別々に収集 ###
    left_hand_data = None
    right_hand_data = None
    if results is not None and results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            cx, cy = get_hand_center(hand_landmarks, img_w, img_h)
            extended_fingers = count_curled_fingers(hand_landmarks)
            
            hand_info = {
                'center': (cx, cy),
                'landmarks': hand_landmarks,
                'extended_fingers': extended_fingers
            }
            
            if hand_label == "Left":
                left_hand_data = hand_info
            elif hand_label == "Right":
                right_hand_data = hand_info

            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ### 変更: 危険判定ロジックを簡略化 ###
    is_dangerous = False
    danger_info = ""

    if knife_boxes_scores and left_hand_data is not None:
        knife_boxes_scores.sort(key=lambda x: x[0], reverse=True)
        best_knife_score, best_knife_box = knife_boxes_scores[0]
        knife_center = get_knife_center(best_knife_box)

        # 左手の状態をチェック
        dist_to_knife = calculate_distance(left_hand_data['center'], knife_center)
        
        # 左手の指が伸びていて、包丁が危険距離内にある場合
        if left_hand_data['extended_fingers'] >= MIN_EXTENDED_FINGERS and dist_to_knife < DANGER_DISTANCE:
            is_dangerous = True
            danger_info = f"Left hand too close! Fingers: {left_hand_data['extended_fingers']}, Dist: {dist_to_knife:.0f}px"
            
            # 危険な左手に赤い円を描画
            cv2.circle(frame, left_hand_data['center'], 60, (0, 0, 255), 4)

        # 包丁の描画
        xmin, ymin, xmax, ymax = best_knife_box
        color = (0, 0, 255) if is_dangerous else (0, 255, 0)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
        score_text = f'Knife: {best_knife_score:.2f}'
        cv2.putText(frame, score_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 危険警告の描画
    if is_dangerous:
        draw_danger_warning(frame)
        if danger_info:
            cv2.putText(frame, danger_info, (10, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ### 変更: 手の情報を左右別に表示 ###
    if left_hand_data:
        info_text = f"Left Hand: {left_hand_data['extended_fingers']}/5 fingers extended"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.circle(frame, left_hand_data['center'], 5, (255, 255, 0), -1)

    if right_hand_data:
        info_text = f"Right Hand: {right_hand_data['extended_fingers']}/5 fingers extended (Holding)"
        cv2.putText(frame, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.circle(frame, right_hand_data['center'], 5, (0, 255, 0), -1)

    if results is not None:
        del results
    gc.collect()

    cv2.imshow("Knife Safety Detection System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()