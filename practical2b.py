import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"C:\Users\student\Desktop\52\cv\pp.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = (1/9) * np.ones((3, 3))

smoothed_image = cv2.filter2D(gray_image, -1, kernel)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(smoothed_image, cmap='gray')
plt.title('Smoothed Image')

plt.show()

cv2.imwrite('smoothed_image.jpg', smoothed_image)
