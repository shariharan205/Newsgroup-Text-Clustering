import numpy as np
import matplotlib.pyplot as plt
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

def breakpoint():
    """

    Allows the flow of program based on the control of the user.
    """
    print "\nContinue? [y/n] : "
    inp = raw_input()

    if inp != 'y':
        exit()



class Clustering(object):

    def __init__(self):

        self.dim_obj_map = {
                "lsi" : TruncatedSVD,
                "nmf" : NMF
        }

        self.max_homogeneity = -float("inf")

    def get_data(self, category_list = [], subset_var = 'train'):
        """
        Takes a category list and a flag whether train/test set has to be retrieved and returns
        the corresponding data from 20newsgroups. If no list is passed, all data is returned.
        """

        if not category_list:
            return fetch_20newsgroups(subset=subset_var, random_state=42, shuffle=True)

        return fetch_20newsgroups(subset=subset_var, categories=category_list, random_state=42, shuffle=True)


    def get_stop_words(self):
        """
        Returns a list of stop words
        """
        return text.ENGLISH_STOP_WORDS #contains 318 stop words


    def get_stemmer(self):
        """
        Snowball stemmer in English performs better than PorterStemmer
        Example : stem("generously") with Porter gives "gener" while SnowballStemmer gives "generous"
        """
        #return PorterStemmer()
        return SnowballStemmer("english")

    def preprocess(self, txt):
        """
        Takes a string and preprocesses it by:
            1. Removing all characters other than alphabets.
            2. Split the string into words
            3. Removing the stop words
            4. Removing the words with length less than 3
            5. Join the list of words back to a sentence.

        Stemming is not done
        """

        txt = re.sub(r'[^a-zA-Z]+', ' ', txt)
        words = txt.split()
        stop_words = self.get_stop_words()
        words = [word.lower() for word in words if word not in stop_words and len(word) > 2]
        return ' '.join(words)
