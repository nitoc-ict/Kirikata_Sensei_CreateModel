from tflite_support import metadata

# TFLITE_MODEL_PATH = "./model1_float16.tflite"    # ご自身のtfliteファイル名に置換
TFLITE_MODEL_PATH = "./model1_float16.tflite"    # ご自身のtfliteファイル名に置換

displayer = metadata.MetadataDisplayer.with_model_file(TFLITE_MODEL_PATH)
print(displayer.get_metadata_json())