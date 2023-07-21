import numpy as np 
import re
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

def bag_of_words(text):
    # CountVectorizer cf. scikit-learn
    vectorizer = CountVectorizer()

    # Fit transform
    vectorizer.fit_transform([text])

    # Get names
    feature_names = vectorizer.get_feature_names_out()

    # Get the bag of words representation as a dictionary 
    bag_of_word = dict(zip(feature_names, vectorizer.transform([text]).toarray()[0]))

    return feature_names


def get_dict():
    my_data = pd.read_csv('data\word-freq-top5000.csv', delimiter=',', usecols=[1]).to_numpy()[:,0]
    return my_data

def to_vectorize(text):
    # CountVectorizer cf. scikit-learn
    vectorizer = CountVectorizer()

    # Fit transform

    print(len(vectorizer.get_feature_names_out()))
        



if __name__ == "__main__":
    testText = "You can train this autoencoder on the Olivetti faces dataset using stochastic gradient descent (SGD) or any other optimization algorithm of your choice. The goal of training is to minimize the reconstruction loss (BCELoss) between the input image and its reconstructed version. Once trained, you can use the encoder to extract features from the Olivetti faces dataset, and the decoder to reconstruct the original images from the encoded features."
    my_data = pd.read_csv('data\word-freq-top5000.csv', delimiter=',', usecols=[1]).to_numpy()[:,0]
    my_data = print(np.unique(my_data).shape)
    to_vectorize(my_data)