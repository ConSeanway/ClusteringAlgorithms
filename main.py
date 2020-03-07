'''
Sean Conway 
CSE 347
3/2/2020
Main method for testing spectral clustering + kmeans
'''
import kmeans
import spectralClustering

# This method is simply an introduction that gets the clustering method for us
def getClusteringMethod():
    print "Welcome to Conway's Clustering Algorithms"
    print "Hope you enjoy!"
    method = raw_input("Would you like to use kmeans (k) or spectral clustering (s)? ")
    # Loop until we get a valid input
    while method != "k" and method != "s":
        method = raw_input("Invalid entry.  Please press k for kmeans clustering or s for spectral clustering. ")
    return method

# This method just asks the user to input which of the two datasets they'd like to use
def getDataset():
    filename = raw_input("Would you like to test iyer.txt or cho.txt? ")
    # Loop until we get a valid input
    while filename != "iyer.txt" and filename != "cho.txt":
        filename = raw_input("Please enter either iyer.txt or cho.txt to proceed: ")
    return filename

# The main method is straightforward.  Get the clustering method and file name from the user, then call the correct clustering algorithm
def main():
    clusteringMethod = getClusteringMethod()
    filename = getDataset()
    if clusteringMethod == "k":
        kmeans.main(filename)
    elif clusteringMethod == "s":
        spectralClustering.main(filename)

if __name__ == '__main__':
    main()
