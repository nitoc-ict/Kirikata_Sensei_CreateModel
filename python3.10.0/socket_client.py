import socketio
import threading
from pynput import keyboard

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY4ZTkzOTIyYzYwNjVlY2Y5ZDMxNmZhNSIsInVzZXJuYW1lIjoic3BlY2lhbCIsImlhdCI6MTc2MDEyMjQxNCwiZXhwIjoxNzYwMTI2MDE0fQ.F0uXBXbEC4_octPNOruL7CuB0DH2ozBuetSpw6dowXo"

sio = socketio.Client()

seat_idx = 0
danger_flag = False

# --- Socket.IO関連のイベントハンドラ ---
@sio.on("connect")
def on_connect():
    print("Connected to server")
    print("Socket ID:", sio.sid)
    print("情報を取得するには 'i' を、終了するには 'q' を押してください。")
    sio.emit("my_event", {"data": "Hello, Server!"})

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

# --- キーボード入力を監視し、処理を分岐させるメイン関数 ---
def on_press(key):
    """キーが押されたときに、Listenerによって直接呼び出される関数"""
    global danger_flag
    try:
        if key.char == 'q':
            print("\n終了キー 'q' が押されました。終了します...")
            if sio.connected:
                sio.disconnect()
            return False

        elif key.char == 'i':
            # サーバーへのemit処理を直接呼び出す
            on_press_get_info()
        
        elif key.char == 'd':
            on_press_danger_alert()

        elif key.char == "s":
            if danger_flag:   
              danger_flag = False
              sio.emit("safeSignal", {
                  "room": "room-101",
                  "userId": sio.sid,
                  "username": "special",
                  "seatIndex": seat_idx
              })
              print("\n's'キーを検出: 安全確認フラグをリセットします。")
              return 0
            else:
              print("\n's'キーを検出: 安全確認フラグは既にリセットされています。")
              return 0
            

    except AttributeError:
        pass  # 特殊キーは無視

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

def on_press_danger_alert():
    global danger_flag
    try:
        if sio.connected:
            print("\n'd'キーを検出: サーバーに危険アラートを送信します。")
            sio.emit("dangerAlert", {
                "room": "room-101",
                "userId": sio.sid,
                "username": "special",
                "seatIndex": seat_idx
            })

            danger_flag = True

            return 0
        else:
            print("サーバーに接続されていません。")
            return 0
    except Exception as e:
        print("error:", e)
        return 0

if __name__ == "__main__":
    socket_thread = threading.Thread(target=start_socketio)
    socket_thread.daemon = True
    socket_thread.start()

    print("キーボードの監視を開始します・・・")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
        
    print("プログラムを終了します。")