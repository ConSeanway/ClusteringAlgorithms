'''
Sean Conway
CSE 347
3/2/2020
K-Means Algorithm Implementation
'''

import random
import math
import numpy as np

# Standardize calculation of dataset values on the normal scale by using Z-scaling
def normalize(dataset):
    dataset_normed = (dataset - dataset.mean(0)) / dataset.std(0)
    return dataset_normed

# Error checking for kmeans inputs
def errorChecking(k, dataset, maxIterations):
    # Error checking for k's value
    if not isinstance(k, (int, long)):
        print "K was not selected to be an integer.  Please try again"
        quit()
    elif k < 1:
        print "Choice of k is less than 1, quitting program"
        quit()
    elif k > len(dataset):
        print "K was chosen to be larger than the dataset, please pick again"
        quit()
        
    # Error checking for the number of max Iterations of kmeans
    if not isinstance(k, (int, long)):
        print "maxIterations was not selected to be an integer.  Please try again"
        quit()
    elif k < 1:
        print "Choice of maxIterations is less than 1, quitting program"
        quit()

# Determines Euclidean distance between 2 data points (attribute vectors)
def findEuclideanDistance(dataPoint1,dataPoint2):
    dist = 0.0
    # Note that this works for n-dimensions due to our "for" loop
    for i in range(0,len(dataPoint1)):
        dist += (dataPoint1[i] - dataPoint2[i])**2
    return math.sqrt(dist)
    
# This is our main kmeans algorithm.  Takes in an integer "k", a set of attribute vectors (the dataset),
# and the maximum number of iterations (also an integer)
def kmeans(k, dataset, maxIterations):
    
    # Error check our parameters (K and maxIterations) to ensure that they are valid 
    errorChecking(k, dataset, maxIterations)
    
    # Set up the cluster arrays that we'll use to assign our data points to clusters
    oldCluster = len(dataset) * [-1]
    cluster = len(dataset) * [0]
    cluster_centers = []
    
    # Initialize the cluster centers to a random data point
    for i in range(0,k):
        cluster_centers += [random.choice(dataset)]
        
    # Reset the value of i, since we use it in the next "for" loop
    i = 0
    
    # This is only really used if we need to recompute the centroids (we assign no data points to a cluster)
    forceRecalculation = False
    
    while ((cluster != oldCluster) and (i < maxIterations)) or (forceRecalculation == True):
        # Assign our "old cluster" our current cluster list
        oldCluster = list(cluster)
        # Reassign the data points to the closest cluster center
        changeClusterAssignment(dataset, cluster, cluster_centers)
        # Update the cluster centroids, and see if we need to recalculate or not
        forceRecalculation = updateClusterPosition(dataset, cluster, cluster_centers)
        i += 1
    
    reportSumSquaredError(dataset, cluster, cluster_centers)
    
    # Print out the results, after reporting the sum squared error
    #print "After " +  str(i) + " iterations, the clusters are: ", cluster_centers
    #print "Clustering Assignment after K-means: ", cluster
    print
    return cluster

# This method just reassigns the clusters, based upon its proximity to each given data point
def changeClusterAssignment(dataset, cluster, cluster_centers):
    # For each point in the dataset...
    for point in range(0,len(dataset)):
            min_dist = float("inf")
            # For each cluster...
            for clust in range(0,len(cluster_centers)):
                # Find the minimum distance between this cluster and each data point.  Assign the data point to the
                # cluster it is closest to
                dist = findEuclideanDistance(dataset[point],cluster_centers[clust])
                if (dist < min_dist):
                    cluster[point] = clust
                    min_dist = dist

# Update each cluster's position based upon their proximity to the cluster centers.  This is done after clusters are reassigned
def updateClusterPosition(dataset, cluster, cluster_centers):
    # For each cluster...
        for k in range(0,len(cluster_centers)):
            new_center = [0] * len(dataset[0])
            members = 0
            # for each point in the dataset...
            for point in range(0,len(dataset)):
                # If the cluster that this point belongs to is currently k...
                if (cluster[point] == k):
                    # Iterate through the dataset and increment the number of datapoints that belong to this center
                    for i in range(0,len(dataset[0])):
                        new_center[i] += dataset[point][i]
                    members += 1
            
            for i in range(0,len(dataset[0])):
                # In this case, our cluster has no data points belonging to it.  We need to reselect a center for the data points
                if members == 0:
                    new_center = random.choice(dataset)
                    print "Re-picking cluster centers..."
                    return True 
                # New center is just the average of the current members (need a float for calculation purposes)
                else: 
                    new_center[i] = new_center[i] / float(members) 
            # Assign the new cluster centers
            cluster_centers[k] = new_center
        return False
               
# Report the sum squared error for kmeans     
def reportSumSquaredError(dataset, cluster, cluster_centers):
    nClust = len(np.unique(cluster))
    arr = [0] * nClust
    SSE = 0
    for i in range(0, len(dataset)):
        for j in range(0, len(dataset[0])):
            lookup = cluster[i]
            arr[lookup] += (dataset[i][j] - cluster_centers[lookup][j])**2
            SSE += (dataset[i][j] - cluster_centers[lookup][j])**2
            
    print "---KMEANS---"
    print "Sum squared error is ", SSE
    print "Broken down by cluster, SSE is: ", arr
                    
# postProcessing is used so that we can compare the ground truth to the result of the kmeans algorithm
# Essentially, what we are trying to do is get an estimate of how well the algorithm is doing
# We do this by building a jagged array, for which each element corresponds to a ground truth cluster
# Each element of this larger array contains a list, which represents the data points that actually belong to that cluster
def postProcessing(groundTruth, kResult):
    # trueClusters is the set of unique numbers listed in the original cluster, and ncluster is the number of original clusters
    nclust = len(np.unique(groundTruth))
    
    # Though this number should be the same for both sets, we take the minimum just to be safe
    numDataPoints = min(len(groundTruth), len(kResult))
    
    # Set up an array for our purity calculations
    realClusterings = [[] for i in range(nclust)]
    for i in range(numDataPoints):
        trueVal = int(groundTruth[i])
        realClusterings[trueVal - 2].append(kResult[i] + 1)
    
    # Looking back, I would have liked to include a confusion matrix, but for this application, we're a bit limited in what we can do
    # The cluster numbers currently don't have any meaning other than to make each cluster distinct from each other
    # If these clusters had meaning, I could make a confusion matrix, which could have been interesting to see which clusters
    # the classifier made the most mistakes on
    print "---POSTPROCESSING---"
    print "Assignments after postProcessing are: ", realClusterings
    
    return realClusterings
    
# Find the purity of each cluster, which we need the mode for.  Since we couldn't use packages, I had to make a manual mode calculation
def findPuritySet(realClusterings, nclust):
    # Create arrays of nclust length to track the mode of each cluster, and calculate purity
    mode = [0] * nclust
    purity = [0] * nclust
    maxCount = [0] * nclust
    
    totMax = 0
    totLength = 0
    
    # For all clusters...
    for i in range(0, nclust):
        # Search for unique values...
        for k in range(1 ,nclust + 1):
            currCount = 0
            # in all cells...
            for j in range(0, len(realClusterings[i])):
                # if it matches the current value, increment the current count
                if int(realClusterings[i][j]) == k:
                    currCount += 1
            # if the current value exceeds the maximum found, then update it.  That'll be the new mode
            if currCount >= maxCount[i]:
                maxCount[i] = currCount
                mode[i] = k
        
        # This computes purity statistics, and enables us to report statistics on our cluster
        if len(realClusterings[i]) != 0:
            purity[i] = 100*maxCount[i]/len(realClusterings[i])
            totMax += 100 * maxCount[i]
            totLength += len(realClusterings[i])
        
    totalPurity = totMax / totLength
        
    print "Mode is " + str(mode) + " for each cluster"
    print "The current purity of each respective cluster assignment is " + str(purity) + " %."
    print "Overall purity of this assignment is " + str(totalPurity) + "%"


def main(filename):
    # Load the current dataset
    dataset = np.loadtxt(fname = filename)
    
    # Create ground truth and training datasets
    groundTruth = dataset[:,1]
    
    # Get rid of unneeded attributes (index and ground truth).  Otherwise the classifier could cheat!
    trainData = np.delete(dataset, obj=[0,1], axis=1)
    
    if filename == "iyer.txt":
        # Delete the first attribute in iyer.txt, as it provides us with no additional information
        trainData = np.delete(trainData, obj=[0], axis=1)
        
        # An issue within iyer is that there is no 0 ground truth cluster, only -1.  To make things consistent, I just made
        # -1 its own cluster, but we just represent it with 0 since it's easier
        for i in range(0, len(groundTruth)):
            if groundTruth[i] < 0:
                groundTruth[i] = 0
    
    # Normalize the data before we begin so that it is standardized (common practice for data mining)
    trainData = normalize(trainData)
    
    # K is the number of clusters.  I select it automatically from the code, but this is something I could also ask the user for
    # Same thing with the maximum number of iterations!
    k = len(np.unique(groundTruth))
    maximumIterations = 100
      
    # Run the kmeans algorithm if that's what we want to do
    kResult = kmeans(k,trainData, maximumIterations)
    
    # Report purity statistics after running the dataset through postProcessing
    if k != len(np.unique(groundTruth)):
        print "Not reporting purity statistics because k selected is different from the clusters found in ground truth"
    else:
        cleanedSet = postProcessing(groundTruth, kResult)
        findPuritySet(cleanedSet, k)