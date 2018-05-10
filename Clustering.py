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


    def collect_data(self, categories = []):
        """
        Given the list of categories, retrieve the newsgroup data.
        """

        news_data = self.get_data(categories, subset_var= 'all')
        return news_data

    def get_tfidf(self, news_data, df = 3):
        """
        Training and testing data are preprocessed.
        Countvectorizer and TfidfTransformer are used to extract TF-IDF features.
        """

        for i in range(len(news_data.data)):
            news_data.data[i] = self.preprocess(news_data.data[i])


        cv = CountVectorizer(min_df=df)
        tf = TfidfTransformer()

        counts = cv.fit_transform(news_data.data)
        tfidf = tf.fit_transform(counts)


        print "Dimension of TF-IDF features with min_df = ", df, " : ", tfidf.shape
        return tfidf


    def kmeans(self, data, features, clusters = 2, threshold = 1):
        """
        Runs k-means on given input to find given number of clusters.
        Data is used to get the target labels
        Features are the actual values upon which K-means is performed.
        Threshold value is used to map higher values of labels to number of clusters.
        """

        print "In K-Means : Input data dimension - ", features.shape

        actual_labels = data.target / threshold #To split the target values into groups

        kmeans_result = KMeans(n_clusters=clusters, random_state=1, init='k-means++').fit(features)
        return self.get_metrics(actual_labels, kmeans_result.labels_), kmeans_result


    def get_metrics(self, actual, predicted):
        """
        Given actual and predicted labels, this method calculates the purity metrics
        """

        measure_scores = {
            "Contingency Matrix" : metrics.confusion_matrix(actual, predicted),
            "Homogeneity" :  metrics.homogeneity_score(actual, predicted),
            "Completeness" : metrics.completeness_score(actual, predicted),
            "V-measure" : metrics.v_measure_score(actual, predicted),
            "Adjusted Rand Score" : metrics.adjusted_rand_score(actual, predicted),
            "Adjusted Mutual Info Score" : metrics.adjusted_mutual_info_score(actual, predicted)
        }


        print "\nContingency Matrix: \n", measure_scores["Contingency Matrix"]
        print "\nHomogeneity : ", measure_scores["Homogeneity"]
        print "Completeness : ", measure_scores["Completeness"]
        print "V-measure : " , measure_scores["V-measure"]
        print "Adjusted Rand score : ", measure_scores["Adjusted Rand Score"]
        print "Adjusted Mutual Info Score : ", measure_scores["Adjusted Mutual Info Score"]

        return measure_scores

