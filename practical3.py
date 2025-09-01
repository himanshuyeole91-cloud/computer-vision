import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

img = cv2.imread(r'C:\Users\student\Desktop\52\cv\pp.jpg')

if len(img.shape) == 3 and img.shape[2] == 3:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    gray_img = img

hog_features, hog_image = hog(
    gray_img,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True,
    block_norm='L2-Hys'
)

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hog_image_rescaled, cmap="gray")
plt.title("HOG Features Visualization")
plt.axis("off")

plt.show()

print(f"Size of HOG feature vector: {len(hog_features)}")
