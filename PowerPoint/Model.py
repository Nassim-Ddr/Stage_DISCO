import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import torchvision.transforms as T
from PIL import Image
from CanvasTools import *


## Classe MODEL
## filename: the file with Pytorch model parameters saved in 
## classe_names: class_label mapping
## process: pre-processing before going into the model
class Model():
    def __init__(self, model, transform_function, classe_names = None, tak2Image2Image = True):
        self.model = model
        self.transform_function = transform_function
        self.classe_names = classe_names
        self.tak2Image2Image = tak2Image2Image
        #self.process = Preprocessing()

    
    # a, b (np.array): precedent state, current state
    # return (prediction, probability of estimate)
    def predict(self, a, b):
        # Pre-processing
        #b = self.process.getOnlyMovingObject(a,b)
        output = self.model(self.input(a,b)) # prediction
        index = output.argmax(1) # index with highest probability
        if self.classe_names is not None: index = self.classe_names[index] # change to class label
        return index, round(output.softmax(dim=1).max().item(),3)
    
    def input(self, a, b):
        s = np.concatenate((a,b), 1) if self.tak2Image2Image else b
        s = Image.fromarray(s)
        return self.transform_function(s).unsqueeze(0)

# Pytorch model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(3136, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 4)

        self.transform = transforms.Compose([ 
            transforms.ToTensor(),
            crop(),
            transforms.Resize((64, 64)),
        ])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = -F.max_pool2d(-x, 2)
        x = F.relu(self.conv2(x))
        x = -F.max_pool2d(-x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet2(nn.Module):
    def __init__(self):
        super(LeNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=8, stride=1, padding="same")
        self.conv2 = nn.Conv2d(4, 8, kernel_size=4, stride=1, padding="same")
        self.conv3 = nn.Conv2d(8, 16, kernel_size=4, stride=1,  padding="same")
        self.fc1 = nn.Linear(2048,1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 4)

        self.process_2image = transforms.Compose([
            transforms.ToTensor(),
            crop_normal(),
            transforms.Resize((64, 128)),
        ])

        self.process_1image = transforms.Compose([
            transforms.ToTensor(),
            crop(),
            transforms.Resize((64, 64)),
        ])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = -F.max_pool2d(-x, 2)
        x = F.relu(self.conv2(x))
        x = -F.max_pool2d(-x, 2)
        x = F.relu(self.conv3(x))
        x = -F.max_pool2d(-x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
# Crop Image: Pre-processing class: go inside pytorch transform.Compose
# Take only the bounding box taking inside all objects in powerpint
class crop(object):
    def __call__(self, img):
        # indexes of non-white pixel
        x1, x2, y1,y2 = find_rect(img)
        return img[:, x1:x2, y1:y2]
    
    def __repr__(self):
        return self.__class__.__name__+'()'
    
def find_rect(x):
    a = np.where(x.mean(0) != 1)
    h,w = x.shape[1:]
    try:
        # Calculate bounding box
        x1,x2,y1,y2 = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])     
        # check if the bounding box is too small
        if x2-x1<64 or y2-y1<64: return 0,h,0,w
        return x1,x2,y1,y2
    except: return 0,h,0,w

class crop_normal(object):
    def __call__(self, img):
        # indexes of non-white pixel
        h,w = img.shape[1:]
        img1, img2 = torch.split(img, w//2, dim=2)
        x11,x12,y11,y12 = find_rect(img1)
        x21,x22,y21,y22 = find_rect(img2)
        x1, y1 = min(x11, x21), min(y11,y21)
        x2, y2 = max(x12, x22), max(y12,y22)
        h,w = img.shape[1:]
        h = h//2
        w = w//2
        s_x = np.arange(x1, x2)
        s_y = np.hstack((np.arange(y1, y2), np.arange(w+y1, w+y2)))
        #print(f'{img.shape = }\n{s_x.max() = }\n{s_y.max() = }')
        return img[:,:,s_y][:,s_x,:]
    
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

    ############## Predict ######################  
    # Return 
    # "Align Copy" 
    # "Copy + Drag"
    # "Premier Plan", "Arriere Plan"
    # "AlignLeft", "AlignTop", "AlignRight", "AlignBottom"
    def predict(self, a, b, eps = 20):
        L = [self.predictForeorBackground, self.predictCopyAlign, self.predictCopyDrag, self.predictAlign]
        for f in L:
            pred = f(a,b, eps)
            if pred is not None: return pred
        return "Rien du Tout"
        
    def predictAlign(self, a, b, eps = 20):
        # Meme nombre d'objets, donc possiblement un déplacement d'objet
        index1, index2, r = self.noChange(a, b, self.pos, eps=20)
        a, b = [self.pos(o) for o in a],  [self.pos(o) for o in b]
        if len(a) == len(b) and len(b)>1 and not r:
            L = None
            A, B = np.array(a), np.array(b)
            # Tri
            A = A[index1]
            B = B[index2] 
            D = np.zeros(4)
            index = np.where(np.abs(A-B).sum(1) != 0)[0] # on cherche l'objet qui s'est déplacé
            for o in B[index]:
                L = np.flip(np.argsort(np.abs(A-B)[index][0]))
                D = np.where(np.abs(B - o) <= eps, 1, 0).sum(0) - 1
                D = D[L]
            index = np.argmax(D)
            if D[index] == 0: return 
            return self.args[L[index]]
        return 
    
    def predictCopyAlign(self, a, b, eps = 20):
        # Meme nombre d'objets, donc possiblement un déplacement d'objet 
        if len(a) == len(b) and len(b)>1:
            A, B = np.array([self.pos(o) for o in a]), np.array([self.pos(o) for o in b])
            index = np.where(np.abs(A-B).sum(1) != 0)[0] # on cherche l'objet qui s'est déplacé
            index = index[0] if len(index)>0 else None
            if index == None: return 
            o = B[index]
            D = np.any(np.where(np.abs(B - o) <= eps, 1, 0), axis=1)
            D[index] = False
            for i in np.where(D)[0]:
                if self.compareApprox(b[index], b[i], eps): 
                    return "Copy + Align"
        elif len(b) == (len(a) + 1):
            B = np.array([self.pos(o) for o in b])
            o = B[-1]
            D = np.any(np.where(np.abs(B - o) <= eps, 1, 0), axis=1)
            D[-1] = False
            for i in np.where(D)[0]:
                if self.compareApprox(b[-1], b[i], eps): 
                    return "Copy + Align"
        return 
    
    def predictCopyDrag(self, a, b, eps = 20):
        if len(a) == len(b):
            B = np.abs([self.state(o) for o in b if not isinstance(o,QRectGroup)])
            A = np.abs([self.state(o) for o in a if not isinstance(o,QRectGroup)])
            if np.abs(B-A).sum() < 1: return
            # Objets non groupés
            B = np.abs([self.stateGroup(o) for o in b if not isinstance(o,QRectGroup)])
            A = np.abs([self.stateGroup(o) for o in a if not isinstance(o,QRectGroup)])
            # # Tri
            # index1, index2 = self.argsort(A, B)
            # A = np.array(A)[index1]
            # B = np.array(B)[index2]
            index = np.where(np.abs(A-B).sum(1) != 0)[0] # on cherche l'objet qui s'est déplacé
            index = index[0] if len(index)>0 else None
            if index is not None:
                B = np.abs([self.state(o) for o in b if not isinstance(o,QRectGroup)])
                o = B[index]
                # on check si y a pas un objet semblable à "o" parmi b
                d = np.all(np.abs(B - o) <= eps, axis=1).sum()
                if d > 1: return "Copy + Drag"
            # Objets groupés
            b = [o for o in b if isinstance(o,QRectGroup)]
            B = [(o.height(), o.width())  for o in b]
            if len(B) > 1:
                for index,o in enumerate(np.abs(B[:-1])):
                    Lindex = np.all(np.abs(B - o) <= eps, axis=1)
                    Lindex[index] = False
                    for i in np.where(Lindex)[0]: 
                        if self.compareGroup(b[index], b[i], eps): return "Copy Group Drag"
        elif len(b) == len(a) + 1:
            if isinstance(b[-1], QRectGroup):
                index = -1
                b = [o for o in b if isinstance(o,QRectGroup)]
                B = [(o.height(), o.width())  for o in b]
                Lindex = np.all(np.abs(B - o) <= eps, axis=1)
                Lindex[index] = False
                for i in np.where(Lindex)[0]: 
                    if self.compareGroup(b[index], b[i], eps): return "Copy Group Drag"
            else:
                B = np.abs([self.state(o) for o in b if not isinstance(o,QRectGroup)])
                o = B[-1]
                # on check si y a pas un objet semblable à "o" parmi b
                d = np.all(np.abs(B - o) <= eps, axis=1).sum() 
                if d > 1: return "Copy + Drag"
        return 

    def predictForeorBackground(self, s1, s2, eps = 20):
        # True pas changement de taille, pas de changement de couleur
        index1, index2, r = self.noChange(s1,s2, self.stateGroup)
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
                if np.abs(R[i]).sum() == 0: return 
                if np.any(x>0) and np.all(x>=0): return "Premier Plan"
                if np.any(x<0) and np.all(x<=0): return "Arriere Plan"
            return 
        return 

    ############# UTILE  ##################
    def pos(self, o):
        return np.array((o.left(), o.top(), o.right(), o.bottom()))

    def state(self, o):
        name = o.__class__.__name__
        if name not in self.D: self.D[name] = len(self.D)
        return (self.D[name]*100, 
                o.height(), o.width(), 
                o.color.red(), o.color.green(), o.color.blue(), 
                o.border_color.red(), o.border_color.green(), o.border_color.blue())
    
    def stateGroup(self, o, ref = (0,0)):
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

    def compareApprox(self, o1, o2, eps=20):
        if o1 == o2: return False
        if o1.__class__.__name__ != o2.__class__.__name__: return False
        if isinstance(o1, QRectGroup): return self.compareGroup(o1, o2, eps)
        o1 = np.array(self.state(o1))
        o2 = np.array(self.state(o2))
        return np.all(np.abs(o1 - o2) <= 10)

    def compareGroup(self, o1, o2, D = None, eps=20):
        if len(o2.objects) != len(o1.objects): return False
        # getState
        top, left = min(o1.top(), o1.bottom()), min(o1.left(), o1.right())
        L1 = [self.stateGroup(o, (top,left)) for o in o1.objects]
        top, left = min(o2.top(), o2.bottom()), min(o2.left(), o2.right())
        L2 = [self.stateGroup(o, (top,left)) for o in o2.objects]
        # Sorting objects
        index1, index2 = self.argsort(L1, L2)
        L1 = np.array(L1)[index1]
        L2 = np.array(L2)[index2]
        return np.all((L1 - L2) <= eps)

    def noChange(self, s1,s2, state_function, eps = 10):
        if len(s1) != len(s2) or len(s1)==0 or len(s2)==0: return None,None,False
        L1 = np.abs([state_function(o) for o in s1])
        L2 = np.abs([state_function(o) for o in s2])
        index1, index2 = self.argsort(L1, L2)
        return index1, index2, np.all((L1[index1] - L2[index2]) <= eps)

            
##################### Train Function #########################
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")       
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")    



