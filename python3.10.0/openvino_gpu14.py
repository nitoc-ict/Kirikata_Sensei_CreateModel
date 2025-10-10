import gc
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

import socketio
import threading
from pynput import keyboard

running = True

# （カメラ設定、YOLOロードなどは変更なし）
# ...
input_length = 640
model_path = "C:/Users/mi241326/idea/procon/createModel/python3.10.0/model/model1/best_openvino_model"
device = "intel:gpu"
task = "detect"
detection_interval = 2
confidence_threshold = 0.50
DANGER_DISTANCE = 50 # 危険距離（ピクセル） 50
MIN_EXTENDED_FINGERS = 2
# ### 変更点：握りこぶし判定用のしきい値を追加 ###
# この値は正規化座標系での距離。0.1〜0.2あたりで調整。
FIST_THRESHOLD = 0.21

# ### 追加：マルチフレーム判定用のパラメータ ###
DANGER_CONFIRMATION_FRAMES = 4  # このフレーム数だけ危険が続いたらアラームON
SAFE_CONFIRMATION_FRAMES = 10   # このフレーム数だけ安全が続いたらアラームOFF

yolo_model = YOLO(model_path, task=task)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_length); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_length)

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY4ZTkzOTIyYzYwNjVlY2Y5ZDMxNmZhNSIsInVzZXJuYW1lIjoic3BlY2lhbCIsImlhdCI6MTc2MDExNTAxNCwiZXhwIjoxNzYwMTE4NjE0fQ.sEoUs3PFj6PW2vKUPzlc9Z_On8NuSbpfRzha7CSJbvo"
sio = socketio.Client()

seat_idx = 0

# --- Socket.IO関連のイベントハンドラ ---
@sio.on("connect")
def on_connect():
    print("Connected to server")
    print("Socket ID:", sio.sid)
    print("情報を取得するには 'i' を、終了するには 'q' を押してください。")

@sio.on("my_response")
def on_message(data):
    print("Message from server:", data)

@sio.on("disconnect")
def on_disconnect():
    print("Disconnected from server")

@sio.on("studentInfo")
def on_student_info(data):
    global seat_idx
    seat_idx = data["occupiedSeats"][0]
    print("座席番号:", seat_idx)
    print("Received student info:", data)

# --- Socket.IOクライアントを起動・実行する関数 ---
def start_socketio():
    """この関数はサブスレッドで実行されます"""
    try:
        sio.connect(
          "http://localhost:3000",
          auth={"token": token}
        )
        sio.emit("join",{
          "role": "special",
          "room": "room-101",
          "username": "special"
        })
        sio.wait()
    except socketio.exceptions.ConnectionError as e:
        if sio.connected:
            print("Connection Error:", e)
    finally:
      if sio.connected:
        sio.disconnect()
      print("Socket.IO client has stopped.")


# 'i'キーが押された時の処理
def on_press_get_info():
    try:
        if sio.connected:
            print("\n'i'キーを検出: サーバーに情報をリクエストします。")
            sio.emit("studentInfo", {
                "room": "room-101",
            })
        else:
            print("サーバーに接続されていません。")
    except Exception as e:
        print("error:", e)
        return 0
    
# --- pynputのキーボード監視関数 ---
def on_press(key):
    global running, cap
    try:
        if key.char == 'q':
            print("終了キー 'q' が押されました。")
            if sio.connected:
                sio.disconnect()
            running = False  # ★ メインループを終了させる
            return False     # リスナー自身を停止させる

        # 'i'や'd'キーの処理もここで行う
        elif key.char == 'i':
            on_press_get_info()

    except AttributeError:
        pass


def on_press_danger_alert():
    try:
        if sio.connected:
            print("\n'd'キーを検出: サーバーに危険アラートを送信します。")
            sio.emit("dangerAlert", {
                "room": "room-101",
                "userId": sio.sid,
                "username": "special",
                "seatIndex": seat_idx
            })

            return 0
        else:
            print("サーバーに接続されていません。")
            return 0
    except Exception as e:
        print("error:", e)
        return 0
    
    
def on_press_safe_signal(): 
    try:
        if sio.connected:
            print("\n's'キーを検出: サーバーに安全確認信号を送信します。")
            sio.emit("safeSignal", {
                "room": "room-101",
                "userId": sio.sid,
                "username": "special",
                "seatIndex": seat_idx
            })

    except Exception as e:
        print("error:", e)


# ### 変更点：ここから新しい判定関数 ###

def is_hand_a_fist(landmarks):
    """手が握りこぶしかを判定する"""
    wrist = landmarks.landmark[0]
    
    # 指先(tip)のランドマークID
    tip_ids = [8, 12, 16]
    
    total_distance = 0
    for tip_id in tip_ids:
        tip = landmarks.landmark[tip_id]
        # 各指先と手首の距離を計算
        distance = np.hypot(tip.x - wrist.x, tip.y - wrist.y)
        total_distance += distance
        
    # 指先と手首の平均距離を算出
    avg_distance = total_distance / len(tip_ids)
    
    # 平均距離がしきい値より小さければ「握りこぶし」と判定
    return avg_distance < FIST_THRESHOLD

def is_finger_extended(landmarks, tip_id, pip_id, mcp_id):
    """【角度判定版】関節の角度で指が伸びているか判定。"""
    tip = np.array([landmarks.landmark[tip_id].x, landmarks.landmark[tip_id].y])
    pip = np.array([landmarks.landmark[pip_id].x, landmarks.landmark[pip_id].y])
    mcp = np.array([landmarks.landmark[mcp_id].x, landmarks.landmark[mcp_id].y])
    v_pip_mcp = mcp - pip
    v_pip_tip = tip - pip
    dot_product = np.dot(v_pip_mcp, v_pip_tip)
    norm_product = np.linalg.norm(v_pip_mcp) * np.linalg.norm(v_pip_tip)
    if norm_product == 0: return False
    cos_theta = dot_product / norm_product
    return cos_theta < -0.94

def count_extended_fingers(hand_landmarks):
    """伸びている指の数を、握りこぶし判定を優先してカウントする"""
    # ### 変更点：最初に握りこぶしかどうかを判定 ###
    if is_hand_a_fist(hand_landmarks):
        # 握りこぶしなら、伸びている指は0本として処理を終了
        return 0

    # 握りこぶしでない場合のみ、角度で各指を判定
    count = 0
    finger_ids = [(8, 6, 5), (12, 10, 9), (16, 14, 13)]
    for tip_id, pip_id, mcp_id in finger_ids:
        if is_finger_extended(hand_landmarks, tip_id, pip_id, mcp_id):
            count += 1
    return count

def get_hand_center(landmarks, width, height):
    xs = [lm.x for lm in landmarks.landmark]; ys = [lm.y for lm in landmarks.landmark]
    return int(np.mean(xs) * width), int(np.mean(ys) * height)
def get_knife_center(box):
    xmin, ymin, xmax, ymax = box; return (xmin + xmax) // 2, (ymin + ymax) // 2
def calculate_distance(point1, point2):
    return np.hypot(point1[0] - point2[0], point1[1] - point2[1])
def draw_danger_warning(frame):
    overlay = frame.copy(); cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    warning_text = "!! DANGER !!"; font = cv2.FONT_HERSHEY_TRIPLEX; font_scale = 2.5; thickness = 5
    text_size, _ = cv2.getTextSize(warning_text, font, font_scale, thickness)
    text_x = (frame.shape[1] - text_size[0]) // 2; text_y = (frame.shape[0] + text_size[1]) // 2
    cv2.putText(frame, warning_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 3)
    cv2.putText(frame, warning_text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)

if __name__ == "__main__":
    socketio_thread = threading.Thread(target=start_socketio, daemon=True)
    socketio_thread.start()

    # 2. キーボード監視スレッドを開始
    keyboard_listener_thread = threading.Thread(
        target=lambda: keyboard.Listener(on_press=on_press).run(),
        daemon=True
    )
    keyboard_listener_thread.start()

    # フレームループ
    frame_count = 0; yolo_results = None

    # ### 追加：ループの外で状態管理用の変数を初期化 ###
    danger_frames_count = 0
    safe_frames_count = 0
    alarm_on = False

    while cap.isOpened() and running:
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
        left_hand_data = None
        right_hand_data = None
        if yolo_results and hasattr(yolo_results, 'boxes') and len(yolo_results.boxes) > 0:
            for box, cls, score in zip(yolo_results.boxes.xyxy, yolo_results.boxes.cls, yolo_results.boxes.conf):
                if int(cls) == 0:
                    xmin, ymin, xmax, ymax = map(int, box)
                    knife_boxes_scores.append((score.item(), [xmin, ymin, xmax, ymax]))

        left_hand_data, right_hand_data = None, None
        if results and results.multi_hand_landmarks:
            for hand_landmarks, handedness_obj in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness_obj.classification[0].label
                cx, cy = get_hand_center(hand_landmarks, img_w, img_h)
                extended_fingers = count_extended_fingers(hand_landmarks)
                hand_info = {'center': (cx, cy), 'landmarks': hand_landmarks, 'extended_fingers': extended_fingers}
                if hand_label == "Left": left_hand_data = hand_info
                elif hand_label == "Right": right_hand_data = hand_info
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        is_dangerous_this_frame = False
        danger_info = ""

        if knife_boxes_scores and left_hand_data is not None:
            knife_boxes_scores.sort(key=lambda x: x[0], reverse=True)
            best_knife_score, best_knife_box = knife_boxes_scores[0]
            knife_center = get_knife_center(best_knife_box)

            if left_hand_data['extended_fingers'] >= MIN_EXTENDED_FINGERS:
                landmarks = left_hand_data['landmarks']
                finger_ids = [(8, 6, 5), (12, 10, 9), (16, 14, 13)]
                for tip_id, pip_id, mcp_id in finger_ids:
                    if is_finger_extended(landmarks, tip_id, pip_id, mcp_id):
                        tip_landmark = landmarks.landmark[tip_id]
                        tip_pos = (int(tip_landmark.x * img_w), int(tip_landmark.y * img_h))
                        dist_tip_to_knife = calculate_distance(tip_pos, knife_center)
                        if dist_tip_to_knife < DANGER_DISTANCE:
                            is_dangerous_this_frame = True
                            danger_info = f"Finger tip too close! Dist: {dist_tip_to_knife:.0f}px"
                            cv2.circle(frame, tip_pos, 15, (0, 0, 255), 3)
                            break
            
            xmin, ymin, xmax, ymax = best_knife_box
            color = (0, 0, 255) if is_dangerous_this_frame else (0, 255, 0)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
            cv2.putText(frame, f'Knife: {best_knife_score:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if is_dangerous_this_frame:
            safe_frames_count = 0
            danger_frames_count += 1
        else: 
            danger_frames_count = 0
            safe_frames_count += 1

        if not alarm_on and danger_frames_count >= DANGER_CONFIRMATION_FRAMES:
            alarm_on = True
            on_press_danger_alert()  # 危険アラートを送信
        elif alarm_on and safe_frames_count >= SAFE_CONFIRMATION_FRAMES:
            alarm_on = False
            on_press_safe_signal()  # 安全確認信号を送信

        if alarm_on:
            draw_danger_warning(frame)
            if danger_info: 
                cv2.putText(frame, danger_info, (10, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if left_hand_data:
            info_text = f"Left Hand: {left_hand_data['extended_fingers']}/3 fingers extended"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if right_hand_data:
            info_text = f"Right Hand: {right_hand_data['extended_fingers']}/3 fingers extended (Holding)"
            cv2.putText(frame, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if results: del results
        gc.collect()

        cv2.imshow("Knife Safety Detection System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

print("クリーンアップ処理を開始します")
cap.release()
cv2.destroyAllWindows()
hands.close()
print("クリーンアップ処理が完了しました")