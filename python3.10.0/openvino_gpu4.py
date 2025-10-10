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

# 危険判定のパラメータ (調整後の値)
DANGER_DISTANCE = 50
MIN_EXTENDED_FINGERS = 3

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
    """指が曲がっているかを判定"""
    tip = landmarks.landmark[finger_tip_id]
    pip = landmarks.landmark[finger_pip_id]
    wrist = landmarks.landmark[0]
    tip_to_wrist = np.hypot(tip.x - wrist.x, tip.y - wrist.y)
    pip_to_wrist = np.hypot(pip.x - wrist.x, pip.y - wrist.y)
    return tip_to_wrist < pip_to_wrist * 1.1

def count_curled_fingers(hand_landmarks):
    """曲がっている指の数をカウント"""
    curled_count = 0
    # 親指
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    index_mcp = hand_landmarks.landmark[5]
    thumb_to_index = np.hypot(thumb_tip.x - index_mcp.x, thumb_tip.y - index_mcp.y)
    thumb_ip_to_index = np.hypot(thumb_ip.x - index_mcp.x, thumb_ip.y - index_mcp.y)
    if thumb_to_index < thumb_ip_to_index * 1.2:
        curled_count += 1
    # 他の指
    if is_finger_curled(hand_landmarks, 8, 6): curled_count += 1
    if is_finger_curled(hand_landmarks, 12, 10): curled_count += 1
    if is_finger_curled(hand_landmarks, 16, 14): curled_count += 1
    if is_finger_curled(hand_landmarks, 20, 18): curled_count += 1
    return 5 - curled_count

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

    frame = cv2.flip(frame, 1)
    frame_count += 1
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = frame.shape

    try:
        results = hands.process(img_rgb)
    except Exception as e:
        print("MediaPipe Handsエラー:", e)
        results = None

    if frame_count % detection_interval == 0:
        try:
            yolo_results = yolo_model(frame, device=device, conf=confidence_threshold)[0]
        except Exception as e:
            print("YOLO推論エラー:", e)

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
            extended_fingers = count_curled_fingers(hand_landmarks)
            hand_info = {'center': (cx, cy), 'landmarks': hand_landmarks, 'extended_fingers': extended_fingers}
            if hand_label == "Left":
                left_hand_data = hand_info
            elif hand_label == "Right":
                right_hand_data = hand_info
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    is_dangerous = False
    danger_info = ""

    if knife_boxes_scores and left_hand_data is not None:
        knife_boxes_scores.sort(key=lambda x: x[0], reverse=True)
        best_knife_score, best_knife_box = knife_boxes_scores[0]
        knife_center = get_knife_center(best_knife_box)

        # ### 変更点：ここから危険判定ロジックを修正 ###
        # まず、左手の指が最低本数以上伸びているかをチェック
        if left_hand_data['extended_fingers'] >= MIN_EXTENDED_FINGERS:
            # 各指先（人差し指から小指）の情報をリスト化
            finger_tip_ids = [
                (8, 6),   # (指先ID, 第二関節ID) for 人差し指
                (12, 10), # for 中指
                (16, 14), # for 薬指
                (20, 18)  # for 小指
            ]
            landmarks = left_hand_data['landmarks']
            
            # 各指先についてループ処理
            for tip_id, pip_id in finger_tip_ids:
                # この指が曲がっていない（伸びている）ことを確認
                if not is_finger_curled(landmarks, tip_id, pip_id):
                    # 指先のスクリーン座標を取得
                    tip_landmark = landmarks.landmark[tip_id]
                    tip_pos = (int(tip_landmark.x * img_w), int(tip_landmark.y * img_h))
                    
                    # 指先と包丁の中心との距離を計算
                    dist_tip_to_knife = calculate_distance(tip_pos, knife_center)
                    
                    # 距離が危険なしきい値より近いか判定
                    if dist_tip_to_knife < DANGER_DISTANCE:
                        is_dangerous = True
                        danger_info = f"Finger tip too close! Dist: {dist_tip_to_knife:.0f}px"
                        
                        # 危険と判断された指先に赤い円を描画
                        cv2.circle(frame, tip_pos, 15, (0, 0, 255), 3)
                        
                        break  # 一本でも危険な指が見つかればループを抜ける
        
        # ### 変更点：ここまで ###

        xmin, ymin, xmax, ymax = best_knife_box
        color = (0, 0, 255) if is_dangerous else (0, 255, 0)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
        cv2.putText(frame, f'Knife: {best_knife_score:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if is_dangerous:
        draw_danger_warning(frame)
        if danger_info:
            cv2.putText(frame, danger_info, (10, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if left_hand_data:
        info_text = f"Left Hand: {left_hand_data['extended_fingers']}/5 fingers extended"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    if right_hand_data:
        info_text = f"Right Hand: {right_hand_data['extended_fingers']}/5 fingers extended (Holding)"
        cv2.putText(frame, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if results:
        del results
    gc.collect()

    cv2.imshow("Knife Safety Detection System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()