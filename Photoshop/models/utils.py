
import os
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import cv2
import win32com.client

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

#Code d'itneraction avec photoshop trouvé ici : https://github.com/kelvin0/ImageAutomation/blob/main/psdbase_utils.py

class PSDBase(object):
    RESIZE_DEFAULT              = 1
    RESIZE_STRETCH_CONSTRAINED  = 2
    def __init__(self,*args,**kwargs):
        self.resize_method = PSDBase.RESIZE_DEFAULT
        
    def open(self,path):
        pass
    def close(self,doc):
        pass
    def compose(self,decorpath,imagepath,target_layername,targetpath,targetname):
        pass
    def shutdown(self):
        pass
        
class Photoshop(PSDBase):
    _ps = None
    SILENT_CLOSE = 2
    def __init__(self):
        super(Photoshop,self).__init__()
        if Photoshop._ps is None:
            Photoshop._ps = win32com.client.Dispatch("Photoshop.Application")
            Photoshop._ps.BringToFront
            Photoshop._ps.DisplayDialogs = 3            # psDisplayNoDialogs
            Photoshop._ps.Preferences.RulerUnits = 1    # psPixels
            
    def open(self,path,open_and_duplicate=True):
        doc = Photoshop._ps.Open(path)
        if open_and_duplicate:
            duplicate = doc.Duplicate()
            doc.Close(Photoshop.SILENT_CLOSE)
            return duplicate
        return doc
        
    def close(self,doc):
        if doc:
            # Close specified document
            doc.Close(Photoshop.SILENT_CLOSE)
            return
        
        # Close all
        for i in range(Photoshop._ps.Documents.Count):
            Photoshop._ps.Documents.Item(i+1).Close(Photoshop.SILENT_CLOSE)
            
    def shutdown(self):            
        try:
            Photoshop._ps.Quit()
        except:
            pass
            
    def export_jpeg(self,doc,savepath,jpeg_filename):        
        exportWebOptions = win32com.client.Dispatch( "Photoshop.ExportOptionsSaveForWeb" )
        #exportWebOptions.Blur
        
        """
        PsColorReductionType
            0 (psPerceptualReduction)
            1 (psSelective)
            2 (psAdaptive)
            3 (psRestrictive)
            4 (psCustomReduction)
            5 (psBlackWhiteReduction)
            6 (psSFWGrayscale)
            7 (psMacintoshColors)
            8 (psWindowsColors)
        """
        #exportWebOptions.ColorReduction          
        #exportWebOptions.Colors
        
        """
        PsDitherType
            1 (psNoDither)
            2 (psDiffusion)
            3 (psPattern)
            4 (psNoise)
        """
        #exportWebOptions.Dither
        #exportWebOptions.DitherAmount
        
        """
        PsSaveDocumentType
            1 (psPhotoshopSave)
            2 (psBMPSave)
            3 (psCompuServeGIFSave)
            4 (psPhotoshopEPSSave)
            6 (psJPEGSave)
            7 (psPCXSave)
            8 (psPhotoshopPDFSave)
            10 (psPICTFileFormatSave)
            12 (psPixarSave)
            13 (psPNGSave)
            14 (psRawSave)
            15 (psScitexCTSave)
            16 (psTargaSave)
            17 (psTIFFSave)
            18 (psPhotoshopDCS_1Save)
            19 (psPhotoshopDCS_2Save)
            25 (psAliasPIXSave)
            26 (psElectricImageSave)
            27 (psPortableBitmapSave)
            28 (psWavefrontRLASave)
            29 (psSGIRGBSave)
            30 (psSoftImageSave)
            31 (psWirelessBitmapSave)
        """
        exportWebOptions.Format = 6
        
        #exportWebOptions.IncludeProfile
        #exportWebOptions.Interlaced
        #exportWebOptions.Lossy
        #exportWebOptions.MatteColor            ==> RGBColor
        #exportWebOptions.Optimized
        #exportWebOptions.PNG8
        
        exportWebOptions.Quality = 72           #(0-100)
        
        #exportWebOptions.Transparency
        #exportWebOptions.TransparencyAmount
        #exportWebOptions.TransparencyDither    ==> PsDitherType
        #exportWebOptions.WebSnap
        
        """
        PsExportType
            1 (psIllustratorPaths)
            2 (psSaveForWeb)
        """
        PsExportType = 2
        
        newfilename = os.path.join(savepath,jpeg_filename)
        doc.Export(newfilename,PsExportType,exportWebOptions)
        
    def save_jpeg(self,doc,savepath,jpeg_filename):
        self._save_psd_to_jpeg(doc,savepath,jpeg_filename)
        
    def _save_psd_to_jpeg(self,doc,savepath,jpeg_filename):
        jpgSaveOptions = win32com.client.Dispatch( "Photoshop.JPEGSaveOptions" )
        jpgSaveOptions.EmbedColorProfile = False
        
        """
        FormatOptions
           1: For most browsers
           2: Optimized color
           3: Progressive"""
        jpgSaveOptions.FormatOptions = 3
        
        """
        PsMatteType
        '   1: No matte
        '   2: PsForegroundColorMatte
        '   3: PsBackgroundColorMatte
        '   4: PsWhiteMatte
        '   5: PsBlackMatte
        '   6: PsSemiGray
        '   7: PsNetscapeGrayMatte"""
        jpgSaveOptions.Matte = 7 
        
        """Quality: 0-12"""
        jpgSaveOptions.Quality = 8
        
        """Scans: 3-5 (Only for FormatOptions=3)"""
        jpgSaveOptions.Scans = 3
        
        """Make up a new name for the new file."""        
        extType = 2 # psLowercase
                
        """Save with new document information and close the file."""
        self._ps.ActiveDocument = doc
        newfilename = os.path.join(savepath,jpeg_filename)
        doc.SaveAs(newfilename, jpgSaveOptions, True, extType)
        
    def _get_target_layer(self,base_psd_folderpath,base_psd_filename):
        base_psd_path = os.path.join(base_psd_folderpath,base_psd_filename)        
        if not os.path.exists(base_psd_path):
            all_base_psd = [f for f in os.listdir(base_psd_folderpath) if f.lower().find('base')>=0]
            if len(all_base_psd) == 0:
                #print "ERROR: Cannot find"
                #print base_psd_path,base_psd_filename
                return None,None
            
            base_psd_filename = all_base_psd[0]
            base_psd_path = os.path.join(base_psd_folderpath,base_psd_filename)        
            if not os.path.exists(base_psd_path):
                #print "ERROR: Cannot find"
                #print base_psd_path,base_psd_filename
                return None,None
        
        base_psd = self.open(base_psd_path,False)
        base_layer = base_psd.ArtLayers.Item(1)
        target_layer = base_layer        
        return base_psd,target_layer
        
    def compose(self,decorpath,imagepath,target_layername,targetpath,targetname):        
        decor_psd = self.open(decorpath)
        
        base_psd_path = os.path.dirname(decorpath)
        base_psd_filename = "{}.psd".format(target_layername)
        base_psd,target_layer = self._get_target_layer(base_psd_path,base_psd_filename)
        if base_psd is None or target_layer is None:            
            #print unicode("ERROR: Cannot find a base layer")
            #print target_layername
            return [None,None]
        
        target_layer_width = target_layer.Bounds[2] - target_layer.Bounds[0]
        target_layer_height = target_layer.Bounds[3] - target_layer.Bounds[1]
        tl_center_x = min(target_layer.Bounds[0],target_layer.Bounds[2]) + (target_layer_width/2)
        tl_center_y = min(target_layer.Bounds[1],target_layer.Bounds[3]) + (target_layer_height/2)
        
        try:
            base_psd.ArtLayers.Item(1).AllLocked = False
            base_psd.ArtLayers.Item(1).Visible = False
        except:
            pass
        
        image = self.open(imagepath)
        if self.resize_method == PSDBase.RESIZE_DEFAULT:
            # Default behavior, simply stretch base image 
            # to fit into layer, regardless of ratio.
            image.ResizeImage(target_layer_width,
                              target_layer_height,
                              decor_psd.Resolution,
                              8) # psAutomatic
        elif self.resize_method == PSDBase.RESIZE_STRETCH_CONSTRAINED:
            # Stretch the base image to become at least as large
            # as the layer, preserving aspect ratio.
            if image.Width < target_layer_width or\
               image.Height < target_layer_height:
                scale_width = target_layer_width/image.Width
                scale_height = target_layer_height/image.Height
                scale = max(scale_width,scale_height)
                image.ResizeImage(image.Width*scale,
                                  image.Height*scale,
                                  decor_psd.Resolution,
                                  8) # psAutomatic
        
        src_layer = image.ArtLayers.Item(1)
        src_layer.Copy()
        self.close(image)
        
        self._ps.ActiveDocument = base_psd
        #base_psd.ArtLayers.Item(1).AllLocked = False 
        pasted_layer = base_psd.Paste()        
        pasted_layer_width = pasted_layer.Bounds[2] - pasted_layer.Bounds[0]
        pasted_layer_height = pasted_layer.Bounds[3] - pasted_layer.Bounds[1]
        pl_center_x = min(pasted_layer.Bounds[0],pasted_layer.Bounds[2]) + (pasted_layer_width/2)
        pl_center_y = min(pasted_layer.Bounds[1],pasted_layer.Bounds[3]) + (pasted_layer_height/2)
        
        pasted_layer.Translate(tl_center_x-pl_center_x,tl_center_y-pl_center_y)
        
        pasted_layer.BlendMode = target_layer.BlendMode
        base_psd.Save() # Saving the base updates the decor, since they are linked!
        
        # Save updated decor as JPEG
        self._save_psd_to_jpeg(decor_psd,targetpath,targetname)
        
        #self.close(decor_psd)
        
        # Make sure our Base.psd remains with only 1 layer, then save it.
        self._ps.ActiveDocument = base_psd
        try:
            base_psd.ArtLayers.RemoveAll()
        except:
            # PS throws an exception when 1 ony layer is left,
            # it looks like it doesnt allow 0 layers... go figure.
            pass
        
        try:
            base_psd.ArtLayers.Item(1).AllLocked = True
            base_psd.ArtLayers.Item(1).Visible = True
        except:
            pass
        base_psd.Save()
        #self.close(base_psd)
        return [base_psd,decor_psd]
