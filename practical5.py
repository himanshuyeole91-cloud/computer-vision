import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
model = YOLO("yolov8n")
from google.colab import files
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
results = model(image_path)
plt.imshow(results[0].plot())
plt.axis("off")
plt.show()