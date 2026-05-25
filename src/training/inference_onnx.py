import os
import onnxruntime as ort
import numpy as np

# 1. Path to your new ONNX model
onnx_model_path = "D:/Projects/real_time_vedio/src/models/lstm-violence-detection.onnx"

# 2. Strict thread limits to prevent CPU saturation
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2  # Limit threads per operation
opts.inter_op_num_threads = 2  # Limit concurrent node threads
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

# 3. Load the session using only the CPU provider
print("🚀 Initializing high-speed ONNX runtime session...")
session = ort.InferenceSession(onnx_model_path, opts, providers=['CPUExecutionProvider'])

# 4. Extract model metadata dynamically
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"📦 Model Input Node Name: '{input_name}'")
print(f"📦 Model Output Node Name: '{output_name}'")

# --- PIPELINE INTEGRATION TEMPLATE ---
# This is how you feed data from your live webcam/video loop:
def predict_violence(sequence_buffer):
    """
    sequence_buffer: A list or array containing exactly 20 frames of features.
    Each frame must contain exactly 34 extracted features.
    """
    # Ensure input data is a 3D numpy array with type float32
    # Shape must be: (1 batch, 20 frames, 34 features)
    input_data = np.array([sequence_buffer], dtype=np.float32)
    
    # Ultra-fast C++ inference execution
    raw_outputs = session.run([output_name], {input_name: input_data})
    
    # Extract prediction value
    prediction = raw_outputs[0][0][0]
    return prediction

# 5. Dummy Verification Run
# Simulating a single sequence match: 1 batch, 20 frames, 34 features
dummy_sequence = np.random.randn(20, 34).astype(np.float32)
violence_score = predict_violence(dummy_sequence)

print(f"\n✅ Test prediction completed successfully!")
print(f"📊 Violence Confidence Score: {violence_score:.4f}")
if violence_score > 0.5:
    print("🚨 Alert: Violence Detected!")
else:
    print("🟢 Status: Normal Activity")
