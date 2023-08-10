
import os
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import cv2

def get_performance(y_true, y_score) : 
    # y_true : list of correct labels (size = n_samples )
    # y_score : probability of predicted classes for the sample (size = n_samples * n_classes)
    
    
    y_pred = [np.argmax(output) for output in y_score]

    accuracy = accuracy_score(y_true, y_pred)

    b_accuracy = balanced_accuracy_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred, average="micro")

    precision = precision_score(y_true, y_pred, average="micro")

    recall = recall_score(y_true, y_pred, average="micro")

    roc_auc = roc_auc_score(y_true, y_score, multi_class="ovr")

    conf_mat = confusion_matrix(y_true, y_pred)


    #Explication + source s: https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    fp = conf_mat.sum(axis=0) - np.diag(conf_mat)  
    fn = conf_mat.sum(axis=1) - np.diag(conf_mat)
    tp = np.diag(conf_mat)
    tn = conf_mat.sum() - (fp + fn + tp)

    


    return [accuracy, f1, precision, recall, roc_auc, conf_mat, fp, fn, tp, tn]

def generate_imgs_dict(path_imgs_org, path_imgs_edt) : 
    list_imgs = os.listdir(path_imgs_org)
    list_imgs_edited = os.listdir(path_imgs_edt)

    d_imgs = dict() 
    d_edited_names = dict()

    for img_name in list_imgs : 
        d_imgs[img_name] = np.array(Image.open(path_imgs_org + img_name))
        
        list_imgs_name_edited = [x for x in list_imgs_edited if x.startswith(img_name[:-4] + "_")]
        d_edited_names[img_name] = list_imgs_name_edited

        for img_name_edited in list_imgs_name_edited : 
            d_imgs[img_name_edited] = np.array(Image.open(path_imgs_edt + img_name_edited))
    
    return d_imgs, d_edited_names
    #d_imgs : dict with key = image name; value = the image as a np.array
    #d_edited_names : dict with key = original image name; value = list of edited image names
        
#Rewrite this with paths in the parameters instead of dictionaries.
def generate_pair_imgs(d_imgs, d_edited_names, save_path) : 
    dataset = []
    for img in d_edited_names : 
        for edited_img in d_edited_names[img] : 
            d = np.hstack((d_imgs[img], d_imgs[edited_img]))
            dataset.append(d)
            i = Image.fromarray(d)
            i.save(save_path + edited_img[:-4] + "_pair.jpg")

def find_edited_areas(image1, image2, threshold=30, min_area=1000):
    diff_image = cv2.absdiff(image1, image2)
    gray_diff = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)
    _, thresholded_diff = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
    bounding_boxes = [cv2.boundingRect(contour) for contour in filtered_contours]
    return bounding_boxes

def bounding_box_from_points(points):
    x_coordinates, y_coordinates = zip(*points)
    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

def find_bb(img_org_name, img_edited_name, path_img_org, path_img_edt, save_folder, min_area=70, threshold=30, scale_factor=1.5) : 
    image1 = cv2.imread(path_img_org + img_org_name)
    image2 = cv2.imread(path_img_edt + img_edited_name)

    # Find the bounding boxes around the edited areas, with a minimum area
    edited_areas = find_edited_areas(image1, image2, threshold=threshold, min_area=min_area)

    # Define the scaling factor (1.0 means no scaling)
    scale_factor = scale_factor

    # Used to define the bounding box of smaller bounding boxes
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
            pilimg = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))

            pilimg.save(save_folder + img_edited_name + "_" + str(i) + ".jpg")


            if (min_x > scaled_x) : min_x = scaled_x
            if (min_y > scaled_y) : min_y = scaled_y
            if (max_x < scaled_x + scaled_w) : max_x = scaled_x + scaled_w 
            if (max_y < scaled_y + scaled_h) : max_y = scaled_y + scaled_h 


    else:
        print("No edited areas found. Try chaning the size.")
        final_image = np.hstack((image1, image2))
        pilimg = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        pilimg.save(save_folder + img_edited_name + "_bb" + ".jpg")
        return 
    
    cropped_image1 = image1[min_y:max_y, min_x:max_x]
    cropped_image2 = image2[min_y:max_y, min_x:max_x]      
    final_image = np.hstack((cropped_image1, cropped_image2))

    pilimg = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    pilimg.save(save_folder + img_edited_name + "_bb" + ".jpg")

def get_bb(img_org, img_edt) : 
    image1 = img_org
    image2 = img_edt

    edited_areas = find_edited_areas(image1, image2, min_area=70)

    scale_factor = 1

    min_x = min_y = 100000
    max_x = max_y = 0

    images_list = []

    if len(edited_areas) > 0:
        for i, (x, y, w, h) in enumerate(edited_areas):
            scaled_w = int(w * scale_factor)
            scaled_h = int(h * scale_factor)
            scaled_x = max(0, int(x - (scaled_w - w) / 2))
            scaled_y = max(0, int(y - (scaled_h - h) / 2))

            cropped_image1 = image1[scaled_y:scaled_y + scaled_h, scaled_x:scaled_x + scaled_w]
            cropped_image2 = image2[scaled_y:scaled_y + scaled_h, scaled_x:scaled_x + scaled_w]
            
            final_image_cropped = np.hstack((cropped_image1, cropped_image2))
            images_list.append(final_image_cropped)
            
            if (min_x > scaled_x) : min_x = scaled_x
            if (min_y > scaled_y) : min_y = scaled_y
            if (max_x < scaled_x + scaled_w) : max_x = scaled_x + scaled_w 
            if (max_y < scaled_y + scaled_h) : max_y = scaled_y + scaled_h 
            
    else:
        return [np.hstack((image1, image2))]

    cropped_image1 = image1[min_y:max_y, min_x:max_x]
    cropped_image2 = image2[min_y:max_y, min_x:max_x]      
    final_image = np.hstack((cropped_image1, cropped_image2))
    images_list.append(final_image)
    return images_list

#Rewrite this with paths in the parameters instead of dictionaries.
def generate_pair_imgs_cropped(d_edited_names, path_imgs_org, path_imgs_edt, save_path, min_area=70, threshold=30, scale_factor=1.5) : 
    for img in d_edited_names : 
        for edited_img in d_edited_names[img] : 
            find_bb(img, edited_img, path_imgs_org, path_imgs_edt, save_path, min_area, threshold, scale_factor)