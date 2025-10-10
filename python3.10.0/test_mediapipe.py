import cv2
import mediapipe as mp

# MediaPipe Handsを初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# カメラを起動
cap = cv2.VideoCapture(0)

print("カメラに向かって手を開いてください。手首(0番)のVisibilityスコアを表示します。")
print("Ctrl+Cで終了します。")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # 処理のために画像を反転・変換
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # MediaPipeで処理
    results = hands.process(image_rgb)

    # 結果を処理
    if results.multi_hand_landmarks:
        # 検出した手から手首のランドマーク（0番）を取得
        wrist_landmark = results.multi_hand_landmarks[0].landmark[0]
        visibility_score = wrist_landmark.visibility
        
        # 1秒ごとくらいにスコアを表示
        print(f"手首のVisibility: {visibility_score:.4f}")

    # 画面表示
    cv2.imshow('MediaPipe Test', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()