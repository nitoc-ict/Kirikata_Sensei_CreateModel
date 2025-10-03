# procon2025

# 環境構築(ver 3.10 を使用)

```shell
# 仮想環境を構築
py -3.10 -m venv venv
# 仮想環境をアクティベート
./venv/Scripts/activate
# pipのアップグレード
python -m pip install --upgrade pip
pip install -r requirements.txt
```

# create_img.py

動画ファイルからフレームごとに画像として保存して、学習データを収集する

# create_model.py

学習データを google colab でファインチューニングを行う。

# detect.py

mediapipe hands と yolo を用いて、実際のモデルの精度の検証のためのデモアプリ

# detect_with_comments.py

detect.pyを少し詳しく説明した。
