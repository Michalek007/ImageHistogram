import cv2
import matplotlib.pyplot as plt

# Load image (change the path to your image)
image = cv2.imread("verybig.png")

# Check if image was loaded
if image is None:
    raise FileNotFoundError("Image not found. Check the file path.")

# Convert BGR (OpenCV default) to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ---------------------------
# Display the image
# ---------------------------
plt.figure(figsize=(6, 4))
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

# ---------------------------
# Grayscale Histogram
# ---------------------------
plt.figure(figsize=(6, 4))
plt.hist(gray.ravel(), bins=256, range=[0, 256])
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

# ---------------------------
# Color Histogram
# ---------------------------
colors = ("r", "g", "b")
plt.figure(figsize=(6, 4))

for i, color in enumerate(colors):
    hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

plt.title("Color Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()
