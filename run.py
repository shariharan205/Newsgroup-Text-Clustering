from Clustering import Clustering

def breakpoint():
    """

    Allows the flow of program based on the control of the user.
    """
    print "\nContinue? [y/n] : "
    inp = raw_input()

    if inp != 'y':
        exit()

print "========================================================================================================\n"

techniques = ["lsi", "nmf"]
ct_ra_categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                    'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

cluster = Clustering()

print "\n=============================================================================================\n"
print "Collecting Computer Technology and Recreational Activity data.........."
ct_ra_data = cluster.collect_data(ct_ra_categories)
print "Performing TF-IDF........"
tfidf = cluster.get_tfidf(ct_ra_data, df=3)
breakpoint()


print "\n=============================================================================================\n"
print "K-Means for k=2 and collecting metrics.........."
cluster.kmeans(ct_ra_data, tfidf, clusters=2, threshold=4)
breakpoint()
