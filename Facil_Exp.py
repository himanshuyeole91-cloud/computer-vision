# ===============================================
# FACIAL EXPRESSION RECOGNITION (Image Only)
# Compatible with Python 3.13
# ===============================================

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
warnings.filterwarnings('ignore')

import cv2

try:
    from fer import FER
except ImportError:
    from fer.fer import FER

# Initialize the FER detector
detector = FER(mtcnn=True)

# === Provide your image path here ===
image_path = "ok.jpg"   # 👈 replace with your image name
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Cannot load image '{image_path}'")
    exit()

# Detect emotions in the image
result = detector.detect_emotions(img)

# Draw results on the image
for face in result:
    (x, y, w, h) = face["box"]
    emotions = face["emotions"]
    top_emotion = max(emotions, key=emotions.get)
    confidence = emotions[top_emotion]

    # Draw face box
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Show detected emotion
    text = f"{top_emotion.capitalize()} ({confidence*100:.1f}%)"
    cv2.putText(img, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Display image
cv2.imshow("Facial Expression Recognition", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print result in console
if result:
    print("Detected emotions:")
    for face in result:
        print(face["emotions"])
else:
    print("No face detected in the image.")
