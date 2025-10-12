import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import os

# 👇 Replace this with your image file path
image_path = r"C:\Users\HP\Desktop\New folder\f1.jpg"

# --- Rest of your code below ---
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis('off')
plt.show()

# Selective search
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
rects = ss.process()

print(f"Total Region Proposals: {len(rects)}")

im_out = image_rgb.copy()
for (x, y, w, h) in rects[:100]:
    cv2.rectangle(im_out, (x, y), (x + w, y + h), (0, 255, 0), 1)

plt.imshow(im_out)
plt.axis('off')
plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.alexnet(pretrained=True).features.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_features(image, rects, model, preprocess, device, max_regions=50):
    features = []
    for (x, y, w, h) in rects[:max_regions]:
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        roi_pil = Image.fromarray(roi)
        input_tensor = preprocess(roi_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model(input_tensor).squeeze()
        features.append(feat.cpu().numpy().flatten())
    return features

features = extract_features(image_rgb, rects, model, preprocess, device)
print(f"Extracted features from {len(features)} region proposals")

labels = [0 if i < len(features)//2 else 1 for i in range(len(features))]
clf = SVC(kernel='linear')
clf.fit(features, labels)

predictions = clf.predict(features)
detected_boxes = [rects[i] for i, pred in enumerate(predictions) if pred == 1]

im_out = image_rgb.copy()
for (x, y, w, h) in detected_boxes:
    cv2.rectangle(im_out, (x, y), (x+w, y+h), (255, 0, 0), 2)

plt.imshow(im_out)
plt.axis('off')
plt.show()
