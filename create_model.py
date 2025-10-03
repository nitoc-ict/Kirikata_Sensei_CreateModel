from ultralytics import YOLO

if __name__ == "__main__":
    # 事前学習済みの軽量モデルをロード（yolo11s系列推奨）
    model = YOLO("yolo11s.pt")

    # Google Colab向け設定例
    results = model.train(
        data='/content/drive/MyDrive/procon/procon2025/model/data2_knife.yaml',  # Colab上のファイルパスに注意
        epochs=30,               # 適切な学習エポック数
        imgsz=640,               # 画像サイズ
        batch=8,                 # ColabのGPUメモリに合わせて調整
        lr0=0.005,               # 初期学習率：やや控えめに
        patience=7,              # 早期停止の猶予
        save_period=10,          # 10エポック毎に保存
        project='knife_detection',
        name='knife_finetune_colab',

        # データ拡張の推奨設定（より多様性を出す）
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=15,
        translate=0.1,
        scale=0.8,
        shear=0.0,
        perspective=0.0,
        flipud=0.1,             # 上下反転少し加える
        fliplr=0.5,             # 左右反転多め
        mosaic=0.7,             # モザイク強めに
        mixup=0.2,              # ミックスアップも適度に

        # 転移学習でバックボーンの最初の10層凍結
        freeze=10,

        val=True,
        plots=True,
        save=True,
        cache=True,             # Colab高速化のためキャッシュ活用

        device='0',             # ColabのGPUを使う設定（'cpu'はNG）
        workers=4,              # Colabで並列ワーク数は多めに

        optimizer='AdamW',
        close_mosaic=15,        # 15エポック目にモザイク終了
        amp=True,               # 自動混合精度ONで高速化かつメモリ節約
        fraction=1.0,
        profile=False,
        verbose=True,
    )

    print("学習完了！")
    print(f"最高精度: {results.results_dict}")

    # 最良モデルのロード
    best_model = YOLO(f'knife_detection/knife_finetune_colab/weights/best.pt')

    # 推論（テスト画像はColabの適切なパスに置く）
    test_results = best_model.predict(
        source='/content/drive/MyDrive/procon/procon2025/model/test_images',
        save=True,
        save_txt=True,
        conf=0.3,
        iou=0.45,
        show_labels=True,
        show_conf=True,
        line_thickness=2,
    )

    # モデル評価
    metrics = best_model.val()
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")

    # tensorflowLite等にモデルエクスポート
    # best_model.export(format='tflite')
