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


for technique in techniques:
    print "Dimension Reduction Technique : ", technique
    args["technique"] = technique
    args["best_dimension"] = best_r[technique]
    cluster.pca(args)
    breakpoint()

print "\n=============================================================================================\n"

print "Transformations...................."

cluster.perform_transformations(args)


print "\n=============================================================================================\n"
print "Experimenting with 20 sub-class labels with clustering............."

print "Collecting all 20 sub classes data.........."
all_data = cluster.collect_data()
print "Performing TF-IDF........"
tfidf = cluster.get_tfidf(all_data, df=3)
breakpoint()


print "Trying out different dimension ranges for 20 clusters"
list_dim_ranges = [[1,2,3,4,5], [10,20,30,40,50,60], [100, 150, 200, 250, 300]]

args = {
         "data" : all_data,
         "features" : tfidf,
         "num_clusters" : 20
       }

best_r = {}

for technique in techniques:
    print "Dimension reduction using - ", technique
    args["technique"] = technique
    args["best_r_value"] = None
    cluster.reset_homogeneity()
    for dimension_range in list_dim_ranges:
        print "Testing for dimensions - ", dimension_range
        args["dimension_range"] = dimension_range
        best_r_val = cluster.dimension_testing(args)
        breakpoint()

    best_r[technique] = best_r_val
    print "Best Dimension found - ", best_r_val, " for technique ", technique


print "\nVisualizing results for 20 clusters......................."

args = {
            "n_clusters" : 20,
            "threshold" : 1,
            "data" : all_data,
            "features" : tfidf,
            "best_r" : best_r
        }


for technique in techniques:
    print "Dimension Reduction Technique : ", technique
    args["technique"] = technique
    args["best_dimension"] = best_r[technique]
    cluster.pca(args)
    breakpoint()


print "Transformations for 20 clusters...................."
cluster.perform_transformations(args)
