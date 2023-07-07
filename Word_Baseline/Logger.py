import numpy as np 
import re

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

    return bag_of_word





if __name__ == "__main__":
    testText = "You can train this autoencoder on the Olivetti faces dataset using stochastic gradient descent (SGD) or any other optimization algorithm of your choice. The goal of training is to minimize the reconstruction loss (BCELoss) between the input image and its reconstructed version. Once trained, you can use the encoder to extract features from the Olivetti faces dataset, and the decoder to reconstruct the original images from the encoded features."
    bow = bag_of_words(testText)
    # Print the bag of words representation
    for word, count in bow.items():
        print(f"{word}: {count}")