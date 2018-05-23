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


print "\n=============================================================================================\n"
print "Plotting the variance retained for 1 to 1000 dimensions in SVD"
cluster.svd_variance_plot(tfidf, dimensions=1000)

print "\n=============================================================================================\n"
print "Plotting metrics for k=2 clusters for different dimensions in SVD and NMF"
dimension_range = [1, 2, 3, 5, 10, 20, 50, 100, 300]
best_r = {}
args = {
         "data" : ct_ra_data,
         "features" : tfidf,
         "dimension_range" : dimension_range,
         "num_clusters" : 2
       }
for technique in techniques:
    print "\nDimension Reduction Technique : ", technique
    args["technique"] = technique
    best_r[technique] = cluster.dimension_testing(args)
    print "\nBest dimension found - ", best_r[technique]
    breakpoint()


print "\n=============================================================================================\n"
print "Visualizing best clustering results using PCA"

args = {
            "n_clusters" : 2,
            "threshold" : 4,
            "data" : ct_ra_data,
            "features" : tfidf
        }