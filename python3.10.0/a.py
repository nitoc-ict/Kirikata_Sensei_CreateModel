import cv2

# for idx in range(5):  # 0～4まで試す
#     cap = cv2.VideoCapture(idx)
#     if cap.isOpened():
#         print(f"カメラ {idx} は利用可能です")
#         cap.release()
#     else:
#         print(f"カメラ {idx} は利用できません")

cap = cv2.VideoCapture(2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 0)  # ミラー表示
    fmrame = cv2.flip(frame, 1)  # 左右反転

    # フレームの処理
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()