from ultralytics import YOLO
import cv2

model = YOLO('yolov8m.pt')  

img = cv2.imread('d:\python cv practicals\IMAGE OBEJCT.jpg')
img = cv2.resize(img, (1280, 720))

results = model(img, conf=0.6, iou=0.5) 

annotated_img = results[0].plot()
cv2.imshow('Improved YOLOv8 Detection', annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
