import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, transforms
import torchvision.transforms as T
from PIL import Image
from CanvasTools import *

class Model():
    def __init__(self, model, classe_names = None):
        self.model = LeNet()
        self.model.load_state_dict(torch.load(model))
        self.classe_names = classe_names
        self.process = Preprocessing()

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            crop(),
            transforms.Resize((64, 64)),
        ])
    
    # a, b (np.array)
    # return (prediction, confiance)
    def predict(self, a, b):
        b = self.process.getOnlyMovingObject(a,b)
        s = Image.fromarray(b)
        x = self.image_transform(s).reshape((1,3,64,64))
        output = self.model(x)
        index = output.argmax(1)
        if self.classe_names is not None:
            index = self.classe_names[index]
        return index, round(output.softmax(dim=1).max().item(),3)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(3136, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 4)

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
    
class crop(object):
    def __call__(self, img):
        a = np.where(img.mean(0) != 1)
        try:
            x1,x2,y1,y2 = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
            h,w = img[:, x1:x2,y1:y2].shape[1:]
            if h<64 or w<64: return img
            return img[:, x1:x2,y1:y2]
        except:
            return img
    
    def __repr__(self):
        return self.__class__.__name__+'()'
    

class Preprocessing():
    def getOnlyMovingObject(self, a, b):
        r = b.copy()
        x,y = np.where(np.mean(a-b, axis=2)<=0)
        r[x,y,:] = 255
        p = 0.8
        return (r*p + b*(1-p)).astype(np.uint8)

class HardCodedModel():
    # a,b : List[QRect]
    # return "R/L/T/B"
    args = ["AlignLeft", "AlignTop", "AlignRight", "AlignBottom"]

    def predict(self, a, b, eps = 20):
        # Meme nombre d'objets, donc possiblement un déplacement d'objet
        a = [(o.left(), o.top(), o.right(), o.bottom()) for o in a]
        b = [(o.left(), o.top(), o.right(), o.bottom()) for o in b]
        if len(a) == len(b) and len(b)>1:
            L = None
            A = np.array(a)
            B = np.array(b)
            D = np.zeros(4)
            index = np.where(np.abs(A-B).sum(1) != 0)[0] # on cherche l'objet qui s'est déplacé
            for o in B[index]:
                L = np.flip(np.argsort(np.abs(A-B)[index][0]))
                D = np.where(np.abs(B - o) <= eps, 1, 0).sum(0) - 1
                D = D[L]
            index = np.argmax(D)
            if D[index] == 0:
                return "Rien du tout"
            return self.args[L[index]]
        return "Rien du tout"
    
    def predictCopyAlign(self, a, b, eps = 20):
        def state(o):
            return (o.height(), o.width(), 
                    o.color.red(), o.color.green(), o.color.blue(), 
                    o.border_color.red(), o.border_color.green(), o.border_color.blue())
        a = [state(o) for o in a if not isinstance(o,QRectGroup)]
        b = [state(o) for o in b if not isinstance(o,QRectGroup)]
        B = np.abs(b)
        for o in B:
            d = np.all(np.abs(B - o) <= eps, axis=1).sum() - 1
            if d > 0:
                return "Copy Align"
        return "Rien Du Tout"
    
# Premier Plan / Arrière Plan
# Copy Style
# EyeDropper
            
            
        



