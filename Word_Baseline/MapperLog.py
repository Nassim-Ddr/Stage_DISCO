import numpy as np 
import re
import pandas as pd
import csv

from sklearn.feature_extraction.text import CountVectorizer



class MapperLog():
    def __init__(self,write = False):
        self.curState=None
        self.vectorizer = CountVectorizer()
        my_data = pd.read_csv('data\word-freq-top5000.csv', delimiter=',', usecols=[1]).to_numpy()[:,0]
        self.vectorizer.fit_transform(my_data)

        # labels = commands
        # textHistory = pair of states
        self.labels = []
        self.textHistory = []

        if write:
            self.file = open("dataTest.csv", 'w', newline='')
            self.writer = csv.writer(self.file)
        self.onWrite = write

    
    # call this func when something is updated into the texteditor
    def update(self,command,state):

        text, cursorPos = state

        treated = self.vectorizer.transform([text]).toarray()[0]
        print(treated)

        # We stack the vector bow with the cursorposition
        treated = np.hstack((treated,np.array([cursorPos])))
        print(treated)

        if self.curState is None :
            self.curState=treated
            return 
        # we store the data that will be used
        if self.onWrite:
            self.writer.writerow(np.hstack([self.curState, treated, command]))

        # change the current data
        self.curState = treated
    

def bag_of_words(text):
    # CountVectorizer cf. scikit-learn
    vectorizer = CountVectorizer()

    # Fit transform
    vectorizer.fit_transform([text])

    # Get names
    feature_names = vectorizer.get_feature_names_out()

    # Get the bag of words representation as a dictionary 
    bag_of_word = dict(zip(feature_names, vectorizer.transform([text]).toarray()[0]))

    return bag_of_word


def get_dict():
    my_data = np.genfromtxt('data\word-freq-top5000.csv', delimiter=',', skip_header=True)[:,2]
    print(my_data)

def to_vectorize(text):
    # CountVectorizer cf. scikit-learn
    vectorizer = CountVectorizer()

    # Fit transform
    vectorizer.fit_transform([text])
    return vectorizer.transform([text])






if __name__ == "__main__":
    testText = "You can train this autoencoder on the Olivetti faces dataset using stochastic gradient descent (SGD) or any other optimization algorithm of your choice. The goal of training is to minimize the reconstruction loss (BCELoss) between the input image and its reconstructed version. Once trained, you can use the encoder to extract features from the Olivetti faces dataset, and the decoder to reconstruct the original images from the encoded features."
    bow = bag_of_words(testText)
    get_dict()