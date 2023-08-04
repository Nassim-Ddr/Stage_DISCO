#CODE TROUVER ICI : https://github.com/kelvin0/ImageAutomation/blob/main/psdbase_utils.py

import os
import sys
import win32com.client
import win32gui
from pyautogui import press, write, hotkey
import random
import time
import numpy as np


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

class Photoshop_Player() : 
    def __init__(self) :
        
        self.input_folder = "C:\\Users\\Nassim\\Desktop\\Stage_DISCO\\Photoshop\\data\\portraits\\"
        self.input_file_list = os.listdir(self.input_folder)
        self.output_folder = "C:\\Users\\Nassim\\Desktop\\Stage_DISCO\\Photoshop\\data\\Player_output\\Params\\"
        self.ps = Photoshop()
        self.handle = win32gui.FindWindow(None, 'Adobe Photoshop (beta)')
        win32gui.SetForegroundWindow(self.handle)

        self.command_dict = {
            "Guassian": ["t", 11, 4, [[1, 1000]]], #param 1 : radius [0.1, 10000]
            "Surface": ["t", 11, 10, [[1, 100], [2, 255]]], #param 1 : radius [1, 100] --- param 2 : Threshold [2, 255]
            "Pinch" : ["t", 13, 1, [[-100, 100]]],  #param 1 : amount  [-100, 100]
            "Twirl" : ["t", 13, 6, [[-999, 999]]],  #param 1 : angle [-999, 999]
            # "Wave" : ["t", 13, 7, [[1, 999], [1, 999], [1, 999], [1, 999], [1, 999], [1, 100], [1, 100]]],   
                                        #param 1 : nb generators [1, 999] --- param 2 : wavelength min = param 3 : wavelengthmax [1, 999]
                                        #param 4 : Amplitude min = param 5 : amplitude max [1,999] ---
                                        #param 6 : Scale horiz = param 7 : Scale verti [1, 100]
            "Sharpen" : ["t", 17, 0, 1], 
            "Diffuse": ["t", 18, 0, 2], 
            "Find edges" : ["t", 18, 3, 1]
        }



        self.command_list = list(self.command_dict.keys())

    def generate_data(self) : 
        for i in range(len(self.input_file_list)) : 
            input_file = self.input_file_list[i]

            for j in range(len(self.command_list)) : 
                
                doc = self.ps.open(self.input_folder + input_file)
                
                commande = self.command_list[j]
                sc = self.command_dict[commande]
                
                l3 = sc[3]
                if (isinstance(l3, int)) : 
                    hotkey("alt", sc[0])
                    press("down", presses=sc[1])
                    press("right")
                    press("down", presses=sc[2])
                    press("enter", presses=sc[3])
                    time.sleep(0.5)
                    self.ps.save_jpeg(doc=doc, savepath=self.output_folder, jpeg_filename=input_file[:-4]+ "_" + commande.lower() +"_out")
                    time.sleep(0.5)
                    self.ps.close(doc)
                else : 
                    l = sc[3]
                    nb_values = 10
                    possible_values = []
                    for rng in l : 
                        possible_values.append(np.linspace(rng[0], rng[1], num=nb_values, dtype=int).tolist())

                    combinaisons = np.array(np.meshgrid(*possible_values)).T.reshape(-1,len(possible_values))
                    combinaisons = [c for c in combinaisons if c[0] < c[1]]

                    for i in range(len(combinaisons)) : 
                        c = combinaisons[i]
                        hotkey("alt", sc[0])
                        press("down", presses=sc[1])
                        press("right")
                        press("down", presses=sc[2])
                        press("enter")
                        for v in c : 
                            press("backspace")
                            write(str(v))
                            press("tab")
                        press("enter")
                        press("enter")
                        time.sleep(0.2)
                
                        self.ps.save_jpeg(doc=doc, savepath=self.output_folder, jpeg_filename=input_file[:-4]+ "_" + commande.lower() + "_" + str(i) +"_out")
                        
                        time.sleep(0.2)
                        self.ps.close(doc)
                        doc = self.ps.open(self.input_folder + input_file)
                        time.sleep(0.2)
                    time.sleep(0.5)
                    self.ps.close(doc)
                    
                
                






if __name__ == '__main__':
    pplayer = Photoshop_Player()
    pplayer.generate_data()
    