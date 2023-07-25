import numpy as np 
import re
import pandas as pd
import csv

from sklearn.feature_extraction.text import CountVectorizer


# Ce logger concerne les classifieurs séparés (avec le classifieur de selection et le classifieur de Bow) ainsi que le classifieur hard codé.
class MapperLog2():
    def __init__(self,write = True,assistant=None):
        self.curState=None
        self.curPosState=None
        self.vectorizer = CountVectorizer()
        #my_data = pd.read_csv('data\word-freq-top5000.csv', delimiter=',', usecols=[1]).to_numpy()[:,0]
        my_data = "Darkness. Just darkness. Darkness not visible. The absence of light. A vacuum I can’t describe. An . . . emptiness.\n\nDarkness in which I can see nothing. Darkness that terrifies me, suffocates me, crushes me. Darkness forced on me whether I like it or not, whether it is daylight or nighttime outside, in which I am expected to sleep. Darkness created by window coverings that cut off light and fresh air, the windows further curtained to prevent stray outside light from entering my room. Darkness.\n\nIn the darkness all I can hear is my clock. And my own heartbeat. And my breathing. At least I am alive. Or am I? It is hard to be sure in the darkness. Darkness, and voices. The voices of my parents, though I am alone, far from them: “Child do this . . . Child do that . . . Child don’t . . . Child why can’t you . . . Child stop . . . Child you must . . . Child, child, child.” Never, ever my name.\n\nLying there, I feel as if I am being forced into a pit, a hole in the ground—being buried, hidden, put away. As if I am disposable. As if my very existence is being denied. As if I must not be seen or heard. As if my birth is a dirty secret, an evil act of mine that must be obliterated without trace. As if I am an object of . . . shame. Why? What could I, a mere child, have done that would cause such a reaction in others, in my father and mother—the man and woman who created me, guardians and enforcers of my darkness?\n\nI am their only child. I think I know why: they never wanted me. I was an accident for them, a mistake they will be careful never to repeat."
        self.vectorizer.fit_transform([my_data])

        # labels = commands
        # textHistory = pair of states
        self.labels = []
        self.textHistory = []
        self.posHistory = []

        if write:
            self.file2 = open("dataSelection.csv", 'w', newline='')
            self.writer2 = csv.writer(self.file2)
            self.file = open("dataBow.csv", 'w', newline='')
            self.writer = csv.writer(self.file)

        self.onWrite = write
        self.assistant = assistant
    
    def reset(self):
        self.curState=None
        self.curPosState=None

    # call this func when something is updated into the texteditor
    def update(self,command,state):

        # format :
        text, cursorPos, cursorStart, cursorEnd, selStart,selEnd = state

        line1, column1 = cursorStart

        line2, column2 = cursorEnd



        treated = self.vectorizer.transform([text]).toarray()[0]
        treatedPos = np.array([cursorPos, line1,column1,line2,column2,selStart,selEnd]).astype(int)
        #print(treated)


        if self.curState is None or self.curPosState is None :
            self.curState=treated
            self.curPosState=treatedPos
            return 
        # we store the data that will be used
        if self.onWrite:
            #print("ok")

            self.writer.writerow(np.hstack([self.curState, treated ,command]))
            self.writer2.writerow(np.hstack([self.curPosState,treatedPos,command]))
            

        # change the current data
        self.curState = treated
        self.curPosState=treatedPos

        if not (self.onWrite):
            #self.assistant.update( (np.hstack((treated, treatedPos)),command) )
            self.assistant.update((treatedPos,command))

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