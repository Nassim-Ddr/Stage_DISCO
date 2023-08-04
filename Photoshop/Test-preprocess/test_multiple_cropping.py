import cv2
import numpy as np

def find_edited_areas(image1, image2, threshold=30, min_area=1000):
    # Compute the absolute difference between the two images
    diff_image = cv2.absdiff(image1, image2)

    # Convert the difference image to grayscale
    gray_diff = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to obtain a binary image
    _, thresholded_diff = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

    # Calculate bounding boxes around the edited areas
    bounding_boxes = [cv2.boundingRect(contour) for contour in filtered_contours]

    return bounding_boxes


def bounding_box_from_points(points):
    x_coordinates, y_coordinates = zip(*points)
    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

def find_bb(img_org, img_edt) : 
    # Load the two images
    image1 = cv2.imread(img_org)
    image2 = cv2.imread(img_edt)

    save_folder = "../data/pair_tmp/"

    # Find the bounding boxes around the edited areas, with a minimum area of 1000 pixels
    edited_areas = find_edited_areas(image1, image2, min_area=70)

    # Define the scaling factor (1.0 means no scaling)
    scale_factor = 1.5

    # Used for the second approach (define the bounding box of smaller bounding boxes)
    min_x = min_y = 100000
    max_x = max_y = 0

    if len(edited_areas) > 0:
        for i, (x, y, w, h) in enumerate(edited_areas):
            # Calculate the scaled dimensions for cropping
            scaled_w = int(w * scale_factor)
            scaled_h = int(h * scale_factor)
            scaled_x = max(0, int(x - (scaled_w - w) / 2))
            scaled_y = max(0, int(y - (scaled_h - h) / 2))

            # Crop the selected area
            cropped_image1 = image1[scaled_y:scaled_y + scaled_h, scaled_x:scaled_x + scaled_w]
            cropped_image2 = image2[scaled_y:scaled_y + scaled_h, scaled_x:scaled_x + scaled_w]
            
            final_image = np.hstack((cropped_image1, cropped_image2))
            cv2.imwrite(save_folder + "final_pair_" + str(i) + ".jpg", final_image)

            if (min_x > scaled_x) : min_x = scaled_x
            if (min_y > scaled_y) : min_y = scaled_y
            if (max_x < scaled_x + scaled_w) : max_x = scaled_x + scaled_w 
            if (max_y < scaled_y + scaled_h) : max_y = scaled_y + scaled_h 

            # Visualize the bounding box for verification
            # cv2.rectangle(image2, (scaled_x, scaled_y), (scaled_x + scaled_w, scaled_y + scaled_h), (0, 255, 0), 2)
            # print("Rectangle N° ", i, " coord (w, h, x, y) : (",  scaled_w, scaled_h, scaled_x, scaled_y, " )")


    else:
        print("No edited areas found. Try chaning the size.")
        final_image = np.hstack((image1, image2))
        cv2.imwrite(save_folder + "final_pair_bb" ".jpg", final_image)
        return 

    #Second approche : Bounding box sur toutes les bounding box
    # cv2.rectangle(image2, (min_x, min_y), (max_x, max_y), (0,0,255), 2)
    cropped_image1 = image1[min_y:max_y, min_x:max_x]
    cropped_image2 = image2[min_y:max_y, min_x:max_x]      
    final_image = np.hstack((cropped_image1, cropped_image2))

    cv2.imwrite(save_folder + "final_pair_bb" ".jpg", final_image)

    # Display the image with bounding boxes (optional)
    cv2.imshow("Cropped around the edits", image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


find_bb("../data/pair_tmp/original.jpg", "../data/pair_tmp/edited_wave.jpg")    

#JOUER AVEC LE SCALE FACTOR
#JOUER AVEC LA TAILLE MIN 
#Discuter idée : data augmentation pour le dataset
#Discuter idée : Modele plus performants
#TRY DEUX APPROCHES : 
    #APPEL MODEL POUR CHAQUE BOUNDING BOX DETECTEE : 
        #pas ouf lol, pas assez de contexte pour le model (testé avec le scale factor = 1)
    #Fusioner les bounding box en une seule grande bounding box et faire une requete au model