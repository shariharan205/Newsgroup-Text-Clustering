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

    def svd_variance_plot(self, features, dimensions = 1000):
        """
        Plot variance retained vs dimensions for SVD for given dimension
        """
        """
        svd = TruncatedSVD(n_components=dimensions)
        svd.fit_transform(features)
        variance_retained = np.cumsum(svd.explained_variance_ratio_)
        plt.plot(np.arange(dimensions), variance_retained*100)
        plt.xlabel('Dimensions', fontsize=15)
        plt.ylabel('Percentage Variance Retained', fontsize=15)
        plt.title('Dimensions vs Percentage Variance Retained for SVD')
        plt.savefig('VarianceRetained.png', format='png')
        plt.show()
        """

        from scipy.sparse.linalg import svds
        u, s, v = svds(features, k=dimensions)

        xxt = np.dot(features, features.T)
        trace_xxt = np.trace(xxt)

        sst = np.dot(s, s.T)
        sst_diag = sst.diagonal()

        a = [ np.sum(sst_diag[:i+1])/float(trace_xxt) for i in range(1001)]
        plt.plot(np.arange(dimensions), a*100)
        plt.xlabel('Dimensions', fontsize=15)
        plt.ylabel('Percentage Variance Retained', fontsize=15)
        plt.title('Dimensions vs Percentage Variance Retained for SVD')
        plt.savefig('VarianceRetained.png', format='png')
        plt.show()



    def reset_homogeneity(self):
        self.max_homogeneity = -float("inf")


    def dimension_testing(self, args):
        """
        For given dimension range, perform Kmeans and plot metrics
        """

        obj = self.dim_obj_map[args["technique"]]
        num_clusters = args["num_clusters"]
        dimension_range = args["dimension_range"]
        technique = args["technique"]
        features = args["features"]
        best_r_val = args.get("best_r_val")

        threshold = 4 if num_clusters == 2 else 1
        homogeneity, completeness, vmeasure, adj_rand, adj_mutual = [], [], [], [], []


        for r in dimension_range:
            print "Number of clusters - ", num_clusters, "  Number of dimensions - ", r
            features_r = obj(n_components = r).fit_transform(features)

            if args.get("tf"):
                for method in args["tf"]:
                    features = getattr(self, method)(features)

            scores, _ = self.kmeans(args["data"], features_r, clusters=num_clusters, threshold=threshold)

            plt.matshow(scores["Contingency Matrix"])
            title = technique + ' num_clusters ' + str(num_clusters) + ' num_dimensions ' + str(r)
            plt.title(title)
            if args.get("tf"):
                title += " " + str(args["tf"])
            plt.savefig(title + '.png', format = 'png')
            plt.show()

            homogeneity.append(scores["Homogeneity"])
            completeness.append(scores["Completeness"])
            vmeasure.append(scores["V-measure"])
            adj_rand.append(scores["Adjusted Rand Score"])
            adj_mutual.append(scores["Adjusted Mutual Info Score"])


        if self.max_homogeneity < np.max(homogeneity):
            best_r_val = dimension_range[np.argmax(homogeneity)]
            self.max_homogeneity = np.max(homogeneity)

        plt.plot(dimension_range, homogeneity, label = "Homogeneity")
        plt.plot(dimension_range, completeness,label = "Completeness")
        plt.plot(dimension_range, vmeasure, label = "VMeasure")
        plt.plot(dimension_range, adj_rand, label = "Adjusted Rand score")
        plt.plot(dimension_range, adj_mutual, label = "Adjusted Mutual Info Score")
        plt.xlabel('Dimensions', fontsize=15)
        plt.ylabel('Purity Measures', fontsize=15)
        if args.get("tf"):
            plt.title(technique + "  " + str(args["tf"]))
        else:
            plt.title('Measures vs #dimensions for ' + technique)
        plt.legend()
        #plt.xscale('log')
        plt.show()
        title = technique + ' num_clusters ' + str(num_clusters) + ' dimensions ' + str(dimension_range) + ' clusters ' + str(num_clusters)
        plt.savefig(title + '.png', format="png")

        return best_r_val

