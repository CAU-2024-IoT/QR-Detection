import cv2
import numpy as np
import pyboof as pb

# Detects all the QR Codes in the image and prints their message and location
data_path = "m2.png"

image_cv = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)

# Optionally scale the image for better detection
scale_factor = 2.3  # Adjust this to the desired scale
image_scaled = cv2.resize(image_cv, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

# Convert OpenCV image to BoofCV format
image_boof = pb.ndarray_to_boof(image_scaled)

# Initialize the QR code detector
detector = pb.FactoryFiducial(np.uint8).microqr()

# Detect QR codes using BoofCV
detector.detect(image_boof)

# Print total number of detected QR codes
print("Detected a total of {} QR Codes".format(len(detector.detections)))

# Iterate through each detected QR code
for qr in detector.detections:
    print("Message: " + qr.message)
    print("     at: " + str(qr.bounds))

    # Convert the Polygon2D to string and clean up the format
    bounds_str = str(qr.bounds)

    # Remove 'Polygon2D(' at the beginning and ')' at the end
    points_str = bounds_str.strip('Polygon2D( ( ) )')

    # Split by ' ) (' to separate the individual points
    points_str = points_str.replace(') (', '),(')
    
    # Split the points by ') (' and then process each point
    points = [tuple(map(float, pt.strip('()').split(','))) for pt in points_str.split('),(')]

    # Convert the points to a numpy array of integers
    points = np.array(points, dtype=np.int32)

    # Reshape the points for OpenCV polylines drawing
    pts = points.reshape((-1, 1, 2))  # Reshape to match the OpenCV format for polylines

#     # Draw the bounding box with OpenCV (red color, thickness 2)
#     cv2.polylines(image_scaled, [pts], isClosed=True, color=(0, 0, 255), thickness=2)  # Red color, thickness 2

# # Show the image with bounding boxes
# cv2.imshow("Detected QR Codes", image_scaled)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
