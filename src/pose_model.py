import os
import shutil
from ultralytics import YOLO

# Set your target directory
target_dir = r"D:\Projects\Distributed_Surviellance_System\src\models"
os.makedirs(target_dir, exist_ok=True)

# 1. Initialize model (will download to current dir if not found)
model = YOLO('yolov8n-pose.pt')

# 2. Export to OpenVINO
print("Starting Export...")
model.export(format='openvino', half=True, imgsz=320, task='pose')

# 3. Define source and destination
src_folder = 'yolov8n-pose_openvino_model'
dst_folder = os.path.join(target_dir, 'yolov8n-pose_openvino_model')

# 4. Move to your Project Directory
if os.path.exists(src_folder):
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder)
    shutil.move(src_folder, dst_folder)
    print(f"✅ Model successfully installed to: {dst_folder}")
