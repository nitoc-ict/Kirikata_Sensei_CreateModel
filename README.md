# procon2025

# 環境構築

- python 3.9.13
- python 3.10.0
  を使用

```shell
# 仮想環境を構築(python 3.9.13)
py -3.9 -m venv venv
# 仮想環境をアクティベート
./venv/Scripts/activate
# pipのアップグレード
python -m pip install --upgrade pip
pip install -r requirements.txt

# 仮想環境を構築(python 3.10.0)
py -3.10 -m venv venv
# 仮想環境をアクティベート
./venv/Scripts/activate
# pipのアップグレード
python -m pip install --upgrade pip
pip install -r requirements.txt
```

# video_frame_extractor.py

動画ファイルからフレームごとに画像として保存して、学習データを収集する

# model_detection_and_inference.py

学習データを google colab でファインチューニングを行う。

# hand_knife_detection.py

mediapipe hands と yolo を用いて、実際のモデルの精度の検証のためのデモアプリ

# hand_knife_detection_with_comments.py

hand_knife_detection.py を少し詳しく説明した。
