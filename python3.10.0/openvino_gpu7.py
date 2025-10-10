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
DANGER_DISTANCE = 50
MIN_EXTENDED_FINGERS = 1

# YOLOモデルのロード
yolo_model = YOLO(model_path, task=task)

# MediaPipe Handsの初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# カメラ設定
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_length)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_length)


def is_finger_extended(landmarks, tip_id, mcp_id):
    """指が伸びているかを判定する最終版ロジック"""
    tip = landmarks.landmark[tip_id]
    mcp = landmarks.landmark[mcp_id]
    
    # 指先(tip)が付け根(mcp)よりY座標が小さい（画面の上にある）なら伸びている
    return tip.y < mcp.y

def count_extended_fingers(hand_landmarks):
    """伸びている指の数をカウントする（人差し指、中指、薬指のみ）"""
    count = 0
    finger_ids = [(8, 5), (12, 9), (16, 13)] # (tip_id, mcp_id)
    for tip_id, mcp_id in finger_ids:
        if is_finger_extended(hand_landmarks, tip_id, mcp_id):
            count += 1
    return count

def get_hand_center(landmarks, width, height):
    """手の中心座標を計算"""
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    return int(np.mean(xs) * width), int(np.mean(ys) * height)

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
    if not ret: break

    frame = cv2.flip(frame, 1)
    frame_count += 1
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = frame.shape

    results = hands.process(img_rgb)

    if frame_count % detection_interval == 0:
        yolo_results = yolo_model(frame, device=device, conf=confidence_threshold)[0]

    knife_boxes_scores = []
    if yolo_results and hasattr(yolo_results, 'boxes') and len(yolo_results.boxes) > 0:
        for box, cls, score in zip(yolo_results.boxes.xyxy, yolo_results.boxes.cls, yolo_results.boxes.conf):
            if int(cls) == 0:
                xmin, ymin, xmax, ymax = map(int, box)
                knife_boxes_scores.append((score.item(), [xmin, ymin, xmax, ymax]))

    left_hand_data, right_hand_data = None, None
    if results and results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            cx, cy = get_hand_center(hand_landmarks, img_w, img_h)
            extended_fingers = count_extended_fingers(hand_landmarks)
            
            hand_info = {'center': (cx, cy), 'landmarks': hand_landmarks, 'extended_fingers': extended_fingers}
            if hand_label == "Left": left_hand_data = hand_info
            elif hand_label == "Right": right_hand_data = hand_info
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    is_dangerous = False
    danger_info = ""

    if knife_boxes_scores and left_hand_data is not None:
        knife_boxes_scores.sort(key=lambda x: x[0], reverse=True)
        best_knife_score, best_knife_box = knife_boxes_scores[0]
        knife_center = get_knife_center(best_knife_box)

        if left_hand_data['extended_fingers'] >= MIN_EXTENDED_FINGERS:
            landmarks = left_hand_data['landmarks']
            finger_ids = [(8, 5), (12, 9), (16, 13)] # (tip_id, mcp_id)
            
            for tip_id, mcp_id in finger_ids:
                if is_finger_extended(landmarks, tip_id, mcp_id):
                    tip_landmark = landmarks.landmark[tip_id]
                    tip_pos = (int(tip_landmark.x * img_w), int(tip_landmark.y * img_h))
                    dist_tip_to_knife = calculate_distance(tip_pos, knife_center)
                    
                    if dist_tip_to_knife < DANGER_DISTANCE:
                        is_dangerous = True
                        danger_info = f"Finger tip too close! Dist: {dist_tip_to_knife:.0f}px"
                        cv2.circle(frame, tip_pos, 15, (0, 0, 255), 3)
                        break
        
        xmin, ymin, xmax, ymax = best_knife_box
        color = (0, 0, 255) if is_dangerous else (0, 255, 0)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
        cv2.putText(frame, f'Knife: {best_knife_score:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if is_dangerous:
        draw_danger_warning(frame)
        if danger_info: cv2.putText(frame, danger_info, (10, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if left_hand_data:
        cv2.putText(frame, f"Left Hand: {left_hand_data['extended_fingers']}/3 fingers extended", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    if right_hand_data:
        cv2.putText(frame, f"Right Hand: {right_hand_data['extended_fingers']}/3 fingers extended (Holding)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if results: del results
    gc.collect()

    cv2.imshow("Knife Safety Detection System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
hands.close()