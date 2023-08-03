#CODE TROUVER ICI : https://github.com/kelvin0/ImageAutomation/blob/main/psdbase_utils.py

import os
import sys
import win32com.client
import win32gui
from pyautogui import press, write, hotkey
import random
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QPushButton, QMessageBox, QWidget, QVBoxLayout
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import transforms
import cv2
import numpy as np
from matplotlib.image import imread

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

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(46656, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Photoshop_Recommander() : 
    def __init__(self) :
        super().__init__()
        print("recommander_launched")
        
        self.ps = Photoshop()
        self.doc = self.ps._ps.ActiveDocument

        #This folder will be used to temporarly save screenshots
        self.tmp_folder = "C:\\Users\\Nassim\\Desktop\\Stage_DISCO\\data\\portraits_tmp\\"

        #This will be used to store the last states of the app
        self.dist_between_states = 5 # En secondes
        self.histo_len = 5
        self.histo_cpt = 1
        self.histo = [] 

        self.ps.save_jpeg(doc=self.doc, savepath=self.tmp_folder, jpeg_filename="tmp_0")
        self.current_state = imread(self.tmp_folder + "tmp_0.jpg")
        
        #Load models
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        
        self.model_normal = LeNet().to(device)
        self.model_normal.load_state_dict(torch.load("../models/model_retrain"))
        
        self.model_crop = LeNet().to(device)
        self.model_crop.load_state_dict(torch.load("../models/model_retrain_crop"))

        

        #Alert to display when we have a recomandation 
        self.alert = QMessageBox()
        self.alert.setWindowTitle("Alert")
        self.alert.setText("The tracked function has been called!")
        
        #
        self.doc = self.ps._ps.ActiveDocument
        self.observe()

    def observe(self) : 
        while(True) : 
            try :
                self.ps.save_jpeg(doc=self.doc, savepath=self.tmp_folder, jpeg_filename="tmp_" + str(self.histo_cpt))
            except : 
                print("Warning : Can't save (App is busy)")
                time.sleep(self.dist_between_states)
                continue
            self.histo.append(self.current_state)
            self.current_state = imread(self.tmp_folder + "tmp_" + str(self.histo_cpt) + ".jpg")
            self.histo_cpt += 1
            self.check_better_command()
            print(self.histo_cpt)
            time.sleep(self.dist_between_states)

    def find_edited_areas(self, image1, image2, threshold=30, min_area=1000):
        diff_image = cv2.absdiff(image1, image2)
        gray_diff = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)
        _, thresholded_diff = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
        bounding_boxes = [cv2.boundingRect(contour) for contour in filtered_contours]
        return bounding_boxes

    def find_bb(self, img_org, img_edt) : 
        image1 = img_org
        image2 = img_edt

        edited_areas = self.find_edited_areas(image1, image2, min_area=10)

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

    def check_better_command(self) : 
        # for old_state in self.histo[:-5] : 
        #     resu = self.ask_model(old_state, self.current_state)
        #     if resu == None : 
        #         continue
        #     else : 
        #         return
        if (len(self.histo) > 2) : 
            self.ask_model(self.histo[:-1][0], self.current_state)

    def ask_model(self, img1, img2) : 
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])
        
        model_inputs = self.find_bb(img1, img2)
        class_small_bb = [0, 0, 0, 0, 0, 0, 0, 0]

        for model_input in model_inputs : 
            image_tensor = transform(Image.fromarray(model_input)).unsqueeze(0)
            image_tensor = image_tensor.cuda()

            with torch.no_grad() : 
                output = self.model_crop(image_tensor)
            _, predicted_class_crop = torch.max(output, 1)
            class_small_bb[predicted_class_crop] += 1
        
        predicted_class_crop = np.argmax(class_small_bb)

        image_tensor = transform(Image.fromarray(np.hstack((img1, img2)))).unsqueeze(0)
        image_tensor = image_tensor.cuda()
        with torch.no_grad() : 
            output = self.model_normal(image_tensor)
        _, predicted_class = torch.max(output, 1)

        if (predicted_class == predicted_class_crop) : 
            print("LES DEUX MODELS SONT D'Accord et il veulent que tu utilise la commande :" + str(predicted_class.item()))
            # self.recomend_command(str(predicted_class.item()))
        else : 
            print("Les deux models ne sont pas d'accord")
            print("model_normal recommande : " + str(predicted_class.item()))
            print("model_crop recommande : " + str(predicted_class_crop.item()))
        
        

    def recomend_command(self, command) : 
        print("recommend command : " + command)

if __name__ == '__main__' : 
    app = QApplication(sys.argv)
    recommander = Photoshop_Recommander()
    sys.exit(app.exec_())