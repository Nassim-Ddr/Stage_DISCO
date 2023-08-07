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
    D = dict()

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
        B = np.abs([self.state(o) for o in b if not isinstance(o,QRectGroup)])
        for o in B:
            d = np.all(np.abs(B - o) <= eps, axis=1).sum() - 1
            if d > 0:
                return "Copy Align"
        return self.compareGroupBetween(a,b, eps)
    
    def compareGroupBetween(self, a, b, eps=20):
        b = [o for o in b if isinstance(o,QRectGroup)]
        B = [(o.height(), o.width())  for o in b]
        print("Filter: ", b)
        if len(B) > 1:
            for index,o in enumerate(np.abs(B[:-1])):
                Lindex = np.all(np.abs(B - o) <= eps, axis=1)
                Lindex[index] = False
                for i in np.where(Lindex)[0]: 
                    if self.compareGroup(b[index], b[i], eps): return "Copy Group Align"
        return "Rien du Tout"
    
    def compareGroup(self, o1, o2, D = None, eps=20):
        if len(o2.objects) != len(o1.objects): return False
        top = min(o1.top(), o1.bottom())
        left = min(o1.left(), o1.right())
        dtype = [(str(i), int) for i in range(11)]
        L1 = [self.stateGroup(o, (top,left)) for o in o1.objects]
        top = min(o2.top(), o2.bottom())
        left = min(o2.left(), o2.right())
        L2 = [self.stateGroup(o, (top,left)) for o in o2.objects]

        #####################
        ##  A Corriger #######
        ######################
        return None
        L1 = np.sort(L1)
        L2 = np.sort(L2)
        return np.all((L1 - L2) <= eps)

    def state(self, o):
        name = o.__class__.__name__
        if name not in self.D: self.D[name] = len(self.D)
        return (self.D[name]*100, 
                o.height(), o.width(), 
                o.color.red(), o.color.green(), o.color.blue(), 
                o.border_color.red(), o.border_color.green(), o.border_color.blue())
    
    def stateGroup(self, o, ref):
        name = o.__class__.__name__
        if name not in self.D: self.D[name] = len(self.D)
        top, left = ref
        return (self.D[name]*100, 
                min(o.top(),o.bottom()) - top, min(o.left(),o.right()) - left,
                o.height(), o.width(), 
                o.color.red(), o.color.green(), o.color.blue(), 
                o.border_color.red(), o.border_color.green(), o.border_color.blue())
    
    def argsort(self, n1, n2):
        dtype = [(str(i),int) for i in range(len(n1[0]))]
        n1 = np.argsort([np.array(tuple(x), dtype=dtype) for x in n1])
        n2 = np.argsort([np.array(tuple(x), dtype=dtype) for x in n2])
        return n1,n2
    
    def predictForeorBackground(self, s1, s2):
        # True pas changement de taille, pas de changement de couleur
        def noChange(s1,s2, eps = 10):
            if len(s1) != len(s2): return None,None,False
            L1 = np.abs([self.state(o) for o in s1])
            L2 = np.abs([self.state(o) for o in s2])
            index1, index2 = self.argsort(L1, L2)
            return index1, index2, np.all((L1[index1] - L2[index2]) <= eps)
        index1, index2, r = noChange(s1,s2)
        if r:
            # relation 1 vs 1
            def f(o1, o2, i,j):
                if i == j: return 0
                v = 0
                if o1.intersects(o2):
                    if i < j: v = -1
                    elif i > j: v = 1
                return v
            # Matrice des relations
            A = np.array([[f(o1, o2, i, j) for j,o2 in enumerate(s1)] for i,o1 in enumerate(s1)])
            B = np.array([[f(o1, o2, i, j) for j,o2 in enumerate(s2)] for i,o1 in enumerate(s2)]) 
            # Tri pour la cohérence
            A = A[index1]
            B = B[index2]
            A = A[:,index1]
            B = B[:,index2]
            R = (B - A)
            index = np.argsort(np.abs(R).sum(1))
            B = B[index][::-1]
            R = R[index][::-1]
            for i,x in enumerate(B):
                if np.abs(R[i]).sum() == 0: return "Rien Du Tout"
                if np.any(x>0) and np.all(x>=0): return "Premier Plan"
                if np.any(x<0) and np.all(x<=0): return "Arriere Plan"
            return "Rien Du Tout"
        return "Rien du Tout"




            

    
# Premier Plan / Arrière Plan
# Copy Style
# EyeDropper
            
            
        



