import os
import sys
import shutil
import tensorflow as tf
import subprocess

# 1. Define file paths
keras_model_path = "D:/Projects/real_time_vedio/src/models/lstm-violence-detection.h5"
onnx_output_path = "D:/Projects/real_time_vedio/src/models/lstm-violence-detection.onnx"
temp_saved_model_dir = "D:/Projects/real_time_vedio/src/models/temp_saved_model"

# 2. Safety check: Verify the Keras model exists
if not os.path.exists(keras_model_path):
    raise FileNotFoundError(f"❌ Could not find Keras model at: {keras_model_path}")

print("🔄 Loading Keras LSTM model...")
model = tf.keras.models.load_model(keras_model_path)

print("💾 Exporting model via Keras 3 Export...")
if os.path.exists(temp_saved_model_dir):
    shutil.rmtree(temp_saved_model_dir)

# Save to temporary folder structure
model.export(temp_saved_model_dir)

print("⚡ Converting SavedModel to ONNX using venv CLI...")
# sys.executable ensures it uses the active (venv) python parser containing tf2onnx
cmd = [
    sys.executable, "-m", "tf2onnx.convert",
    "--saved-model", temp_saved_model_dir,
    "--output", onnx_output_path,
    "--opset", "13"
]

# Execute conversion command line utility securely via local python session
result = subprocess.run(cmd, capture_output=True, text=True)

# 4. Clean up the temporary folder
if os.path.exists(temp_saved_model_dir):
    shutil.rmtree(temp_saved_model_dir)

# 5. Check if the conversion succeeded
if result.returncode == 0:
    print(f"✅ ONNX model successfully saved to: {os.path.abspath(onnx_output_path)}")
else:
    print("❌ ONNX Conversion Failed via command line.")
    print("Error Details:\n", result.stderr)
