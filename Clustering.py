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
