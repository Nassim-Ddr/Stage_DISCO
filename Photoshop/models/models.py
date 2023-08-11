import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as nnf
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms

# TODO : 
#   Add functions to test single images

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

def load_LeNet(dict_path) : 
    model = LeNet()
    model.load_state_dict(torch.load(dict_path))
    return model

def LeNet_Preprocess() : 
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess 

def ResNet() : 
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    num_classes = 8
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1000),  # New layer with 1000 outputs (matches pre-trained model)
        nn.ReLU(),                       # Activation function (you can use others if needed)
        nn.Linear(1000, num_classes)     # Final layer with 8 outputs for your classification task
    )
    model = model.cuda()

    return model

def load_ResNet(dict_path) : 
    model = ResNet()
    model.load_state_dict(torch.load(dict_path))
    return model

def ResNet_Preprocess() : 
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    return preprocess

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) # used for the loss print (useless if not printing) 
    
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
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_basic(dataloader, model):
    model.eval()
    y_true = []
    y_score = []

    with torch.no_grad():
        for X, y in dataloader:
            Xd, yd = X.to(device), y.to(device)
            
            with torch.no_grad() : 
                score = model(Xd)

            
            score_m = nnf.softmax(score, dim=1)
            
            y_score.extend(score_m.tolist())
            y_true.extend(yd.tolist())

    return y_true, y_score
    
def test_basic_threshold(dataloader, model, threshold) : 
    y_true = []
    y_pred = []

    for X, y in dataloader:
        y_true.extend(y.data.cpu().numpy())
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            pred = model(X)
        pred = pred.softmax(dim=1)
        y_pred.extend(pred.data.cpu().numpy())
        
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    indices = y_pred.max(1) > threshold
    y_pred = y_pred[indices].argmax(1)
    y_true = y_true[indices]

    return y_true, y_pred

def test_handmade(dataloader, model):
    #There is one difference between this and the test_basic : 
    #   Since we don't have much handmade data on the filter 0 and 1 (because they take too much to reproduce (about 5 to 10 minute each))
    #   We add one instance of correct prediction so the number of labels is consistent
    
    
    model.eval()
    y_true = [0, 1]
    y_score = [[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0]]

    with torch.no_grad():
        for X, y in dataloader:
            Xd, yd = X.to(device), y.to(device)
            with torch.no_grad() : 
                score = model(Xd)

            
            score_m = nnf.softmax(score, dim=1)
            
            y_score.extend(score_m.tolist())
            y_true.extend(yd.tolist())

    return y_true, y_score

def test_resnet_3(dataloader, model_1, model_2, model_3) : 
    model_1.eval()
    model_2.eval()
    model_3.eval()
    
    y_true = [0, 1]
    y_score = [[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0]]

    for x, y in dataloader : 
        xd, yd = x.to(device), y.to(device)
        
        with torch.no_grad() : 
            score_1 = model_1(xd)
            score_2 = model_2(xd)
            score_3 = model_3(xd)

        score_m = score_1 + score_2 + score_3
        score_m = nnf.softmax(score_m, dim=1)
        
        y_score.extend(score_m.tolist())
        y_true.extend(yd.tolist())

    return y_true, y_score




