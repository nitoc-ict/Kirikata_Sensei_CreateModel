import cv2  # OpenCVライブラリのインポート
import os   # OS操作用ライブラリのインポート

# 現在のスクリプトのディレクトリを取得
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 入力動画ファイルのパスと画像保存先ディレクトリのパスを設定
video_filename = 'video.mp4'  # 入力動画ファイル名
video_path = os.path.join(BASE_DIR, 'public', video_filename)  # 入力動画ファイルのパス
output_dir = os.path.join(BASE_DIR, 'output', os.path.splitext(video_filename)[0])  # 画像保存先

# 保存先ディレクトリが存在しない場合は新たに作成する
os.makedirs(output_dir, exist_ok=True)

# 動画ファイルを読み込むためのVideoCaptureオブジェクトを作成
cap = cv2.VideoCapture(video_path)

# 動画のフレームレート（FPS）を取得
fps = cap.get(cv2.CAP_PROP_FPS)

# 10秒ごとにフレームを保存するための間隔（フレーム数）を計算
ineterval = int(fps * 0.5)  # 0.5秒ごとにフレームを保存

frame_count = 0   # 現在のフレーム番号
saved_count = 0   # 保存した画像の枚数

# 動画の全フレームを順に処理
while True:
    ret, frame = cap.read()  # 1フレーム読み込み
    if not ret:              # フレームが取得できなければ終了
        break
    # 指定した間隔ごとにフレーム画像を保存
    if frame_count % ineterval == 0:
        # 画像ファイル名は<filename>_000.jpgのように連番で保存
        cv2.imwrite(f'{output_dir}/{os.path.splitext(video_filename)[0]}_{saved_count:03d}.jpg', frame)
        saved_count += 1
    frame_count += 1  # フレーム番号をインクリメント

# 動画ファイルを閉じる
cap.release()
# 保存完了メッセージを出力
print(f'保存完了: {saved_count}枚')
