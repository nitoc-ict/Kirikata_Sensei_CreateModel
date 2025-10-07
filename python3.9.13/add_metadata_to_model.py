import yaml
from tflite_support.metadata_writers import writer_utils
from tflite_support.metadata_writers import object_detector
from tflite_support import metadata

# ファイルパス指定
TFLITE_MODEL_PATH = "C:/Users/mi241326/idea/procon/python_venv/model/model/best_float32.tflite"     # ご自身のtfliteファイル名に置換
METADATA_YAML_PATH = "metadata.yaml"          # 添付ファイル名を指定
LABELS_PATH = "labels.txt"                      # 後で作成
OUTPUT_MODEL_PATH = "model_with_metadata.tflite"

# 1. metadata.yamlを読み込む
with open(METADATA_YAML_PATH, 'r', encoding='utf-8') as f:
    metadata_dict = yaml.safe_load(f)

print(metadata_dict.get('names'))


# 2. names情報からlabels.txtを作成
names = metadata_dict.get('names', {})
sorted_keys = sorted(names.keys())
sorted_names = [names[k] for k in sorted_keys]
with open(LABELS_PATH, 'w', encoding='utf-8') as f:
    for name in sorted_names:
        f.write(name + '\n')
print(f"labels.txtを生成しました: {LABELS_PATH}")

# 3. tfliteファイルにメタデータを書き込む
input_model_buffer = writer_utils.load_file(TFLITE_MODEL_PATH)

writer = object_detector.MetadataWriter.create_for_inference(
    input_model_buffer,
    input_norm_mean = [0.0],
    input_norm_std = [1.0],
    label_file_paths = [LABELS_PATH]
)

output_model_buffer = writer.populate()

# 4. メタデータ付加済みモデルファイルを保存
with open(OUTPUT_MODEL_PATH, 'wb') as f:
    f.write(output_model_buffer)

print(f"メタデータを埋め込んだtfliteモデルを保存しました: {OUTPUT_MODEL_PATH}")

# 5. 埋め込み内容の確認
displayer = metadata.MetadataDisplayer.with_model_file(OUTPUT_MODEL_PATH)
print("埋め込まれたメタデータ情報（JSON形式）:")
print(displayer.get_metadata_json())

print("関連ファイルリスト:")
print(displayer.get_packed_associated_file_list())
