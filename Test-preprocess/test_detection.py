import cv2
import numpy as np

# Load the two images
image1 = cv2.imread("../data/pair_tmp/original.jpg")
image2 = cv2.imread("../data/pair_tmp/edited_wave.jpg")

# Compute the absolute difference between the two images
diff_image = cv2.absdiff(image1, image2)

# Convert the difference image to grayscale
gray_diff = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to obtain a binary image
_, thresholded_diff = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate bounding boxes of the contours
bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

# Crop the original image around the edited areas
for bbox in bounding_boxes:
    x, y, w, h = bbox
    cropped_image = image2[y:y + h, x:x + w]
    # Now you can save or display the cropped_image

# Optionally, you can visualize the contours and cropped areas for verification
cv2.drawContours(image2, contours, -1, (0, 255, 0), 2)
cv2.imshow("Contours and cropped areas", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()