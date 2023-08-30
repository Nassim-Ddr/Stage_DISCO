
import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication
from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np
from matplotlib.image import imread
from Recommender_UI import *
from models.utils import *
from models.models import *

class Photoshop_Recommander(QMainWindow) : 
    def __init__(self, model, preprocess) :
        super().__init__()
        print("recommander_launched")
        
        self.ps = Photoshop()
        self.doc = self.ps._ps.ActiveDocument

        #This folder will be used to temporarly save screenshots
        self.tmp_folder = "C:\\Users\\Nassim\\Desktop\\Stage_DISCO\\Photoshop\\data\\portraits_tmp\\"

        #This will be used to store the last states of the app
        self.dist_between_states = 5 # En secondes
        self.histo_len = 5
        self.histo_cpt = 1
        self.histo = [] 

        # #Init the states
        self.ps.save_jpeg(doc=self.doc, savepath=self.tmp_folder, jpeg_filename="tmp_0")
        self.current_state = imread(self.tmp_folder + "tmp_0.jpg")
        self.old_state = None


        #Init the models
        self.model = model
        self.preprocess = preprocess
        
        self.command_labels = ["Diffuse Filter", "Edges Filter", "Gaussian filter", "Pinch Filter", "Sharpen Filter", "Surface Filter", "Twirl Filter", "Wave Filter"]

        #Alert to display when we have a recomandation 
        self.alert = Recommender_UI()
        self.alert.show()
        
        #Init photoshop variables
        self.doc = self.ps._ps.ActiveDocument

        self.timer = QTimer(self)
        self.timer.setInterval(self.dist_between_states*1000)
        self.timer.timeout.connect(self.observe)
        self.timer.start()

    def observe(self) :
        try: 
            self.ps.save_jpeg(doc=self.doc, savepath=self.tmp_folder, jpeg_filename="tmp_" + str(self.histo_cpt))
        except : 
            print("Warning : Can't save (App is busy)")
            # time.sleep(self.dist_between_states)
            return 
        
        self.histo.append(self.current_state)
        self.old_state = self.current_state
        self.current_state = imread(self.tmp_folder + "tmp_" + str(self.histo_cpt) + ".jpg")
        self.histo_cpt += 1
        print(self.histo_cpt)

        if ((self.old_state - self.current_state).sum() < 0.001 ) : 
            print("Warning : No change detected")
            return
        
        self.check_better_command()
        
        

    def check_better_command(self) : 
        # for old_state in self.histo[:-5] : 
        #     resu = self.ask_model(old_state, self.current_state)
        #     if resu == None : 
        #         continue
        #     else : 
        #         return
        if (len(self.histo) > 2) : 
            self.ask_normal_model(self.old_state, self.current_state)

    def ask_normal_model(self, img1, img2) : 
        image_tensor = self.preprocess(Image.fromarray(np.hstack((img1, img2)))).unsqueeze(0)
        # Image._show(T.ToPILImage()(image_tensor[0]))
        with torch.no_grad() : 
            output = self.model(image_tensor)
        output = nnf.softmax(output, dim=1)
        confidence , predicted_class = torch.max(output, 1)
        
        self.recomend_command(self.command_labels[predicted_class.item()], confidence.item())

    def recomend_command(self, command, confidence) : 
        self.alert.update("recommend command : " + command + "\n" + "confidence : "+ f'{(confidence*100):.2f} %')

if __name__ == '__main__' : 

    model = load_LeNet("./models/model_retrain")
    preprocess = LeNet_Preprocess()

    app = QApplication(sys.argv)
    recommander = Photoshop_Recommander(model, preprocess)
    # recommander = Photoshop_Recommander(None, None)
    recommander.observe()

    sys.exit(app.exec_())