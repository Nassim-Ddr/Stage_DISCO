import cv2
import numpy as np

def find_edited_area(image1, image2, threshold=30):
    # Compute the absolute difference between the two images
    diff_image = cv2.absdiff(image1, image2)

    # Convert the difference image to grayscale
    gray_diff = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to obtain a binary image
    _, thresholded_diff = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the bounding box around the edited area
    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return x, y, w, h
    else:
        return None

# Load the two images
image1 = cv2.imread("../data/pair_tmp/original.jpg")
image2 = cv2.imread("../data/pair_tmp/edited_wave.jpg")

# Find the bounding box around the edited area
edit_bbox = find_edited_area(image1, image2)

if edit_bbox is not None:
    x, y, w, h = edit_bbox
    cropped_image = image2[y:y + h, x:x + w]
    # Now you can save or display the cropped_image
else:
    print("No edited area found.")

# Optionally, you can visualize the bounding box for verification
if edit_bbox is not None:
    cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Cropped around the edit", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



