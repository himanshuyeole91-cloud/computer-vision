import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r'C:\Users\student\Desktop\52\cv\pp.jpg')

if len(img.shape) == 3 and img.shape[2] == 3:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    gray_img = img

sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray_img, None)

sift_img = cv2.drawKeypoints(gray_img, keypoints[:50], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(sift_img, cmap="gray")
plt.title("SIFT Keypoints")
plt.axis("off")
plt.show()

print("Information about the first detected SIFT poin
