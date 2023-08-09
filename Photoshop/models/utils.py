import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
from scipy.io.matlab.mio import loadmat, savemat
import numpy as np
from PIL import Image
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as nnf
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt


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






    
# image, labels = dataset[0]
# image = image.to(device)
# image = image.unsqueeze(0)

# score = model(image)[0].cpu().detach().numpy()
# print(score)