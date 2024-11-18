import cv2
import numpy as np

# Load the image
image = cv2.imread('micro.png')

# Resize the image to increase the size of QR codes
scale_percent = 120  # Scale by 300%
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
image_resized = cv2.resize(image, dim)

# gray_resized = image_resized

# Convert resized image to grayscale
gray_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

# # Apply adaptive thresholding for noise reduction
# gray_resized = cv2.adaptiveThreshold(gray_resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# # Apply bilateral filtering to reduce noise while keeping edges sharp
# gray_resized = cv2.bilateralFilter(gray_resized, 9, 75, 75)

# # Apply morphological transformations to remove small noise and enhance QR code structure
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# gray_resized = cv2.morphologyEx(gray_resized, cv2.MORPH_CLOSE, kernel)

# Detect QR codes in the resized image
qr_detector = cv2.QRCodeDetector()
success, data, points, straight_qrcode = qr_detector.detectAndDecodeMulti(gray_resized)

# Process QR codes if found
if points is not None:
    for i, point in enumerate(points):
        qr_points = point.reshape((-1, 2)).astype(int)
        for j in range(len(qr_points)):
            cv2.line(image_resized, tuple(qr_points[j]), tuple(qr_points[(j + 1) % 4]), (0, 255, 0), 2)
        print(f"QR Code {i+1}: Data: {data[i]}, Position: {qr_points}")
else:
    print("No QR codes found.")

# Display result
cv2.imshow('QR Codes', image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
