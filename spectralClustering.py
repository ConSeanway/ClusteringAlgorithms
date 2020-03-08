'''
Sean Conway
CSE 347
3/2/2020
Spectral Clustering Algorithm Implementation
'''
from sklearn import neighbors as nb
from kmeans import kmeans
from kmeans import postProcessing
from kmeans import findPuritySet
from kmeans import normalize
import numpy as np

'''
General Procedure:
1. Represent data points as a symmetric similarity graph
2. Computer the graph Laplacian: L = D - W
3. Compute K eigenvectors corresponding to the K smallest non-zero eigenvalues of L
4. Cluster the eigenvectors with K-means algorithm into K clusters
'''

# Create an adjacency graph of k-nearest neighbors
def createGraph(dataset, k):
    # Find the KNN graph for each 
    graph = nb.kneighbors_graph(dataset, n_neighbors=k, metric='euclidean').toarray()
    # Forces symmetry in the graph
    graph = 0.5*(graph + graph.T)
    # Print the graph
    #print "---Graph---"
    #print graph
    return graph
    
def computeGraphLaplacian(A):
    # Find the graph laplacian
    D = np.diag(A.sum(axis=1))
    laplacian = D - A
    
    #print "---Graph Laplacian---"
    #print laplacian
    return laplacian

# Returns the eigenvalues and eigenvectors for the laplacian matrix
def findEigenValueVectors(laplacian):
    eigenvalues, eigenvectors = np.linalg.eig(laplacian)
    #print 
    #print "---EigenValues---"
    #print eigenvalues
    #print "---EigenVectors---"
    #print eigenvectors
    return eigenvalues, eigenvectors

def clusterEigenvectors(k, laplacian, maxIterations):
    #print
    #print "---Eigenvector Clustering---"
        
    # Call kmeans to cluster the resulting eigenvectors
    clusters = kmeans(k, laplacian, maxIterations)
    return clusters

def main(filename):
    # Load the current dataset
    dataset = np.loadtxt(fname = filename)
    # Create ground truth and training datasets
    groundTruth = dataset[:,1]
    
    # Get rid of unneeded attributes (index and ground truth).  Otherwise, we're cheating our performance metrics
    trainData = np.delete(dataset, obj=[0,1], axis=1)
        
    # Decided to exclude the third column in iyer.txt, since it doesn't give us any more info than we already have
    if filename == "iyer.txt":
        # Delete the column that gives us no extra data
        trainData = np.delete(trainData, obj=[0], axis=1)
        for i in range(0, len(groundTruth)):
            if groundTruth[i] < 0:
                groundTruth[i] = 0
        # These are 2 outlier points that really mess with the clustering in iyer...  Decided to exclude them
        for i in range(len(trainData[0])):
            trainData[362][i] = np.mean(trainData[i], axis=0)
            trainData[490][i] = np.mean(trainData[i], axis=0)
    
    # Normalize the data before we begin so that it is standardized (common practice for data mining)
    trainData = normalize(trainData)
    
    # Select a value for k to test
    k = len(np.unique(groundTruth))
    
    # Follow the spectral clustering procedure of creating a graph, finding the laplacian, computing eigenvectors, and clustering them
    graph = createGraph(trainData, k)
    laplacianMatrix = computeGraphLaplacian(graph)
    eigenvalues, eigenvectors = findEigenValueVectors(laplacianMatrix)
    clusterings = clusterEigenvectors(k, eigenvectors, 100)
    
    # Report purity statistics
    if k != len(np.unique(groundTruth)):
        print "Not reporting purity statistics because k selected > clusters in ground truth"
    else:
        cleanedSet = postProcessing(groundTruth, clusterings)
        findPuritySet(cleanedSet, k)