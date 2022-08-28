import numpy as np
from numpy.core.fromnumeric import argmin

from cluster import cluster


class KMeansClustering:
    """
    K-Means Clustering Model

    Args:
        n_clusters: Number of clusters(int)
    """

    def __init__(self, n_clusters, max_iter=10, delta=0.001):

        self.n_cluster = n_clusters
        self.max_iter = max_iter
        self.delta = delta
        self.centroids = None

    def init_centroids(self, data):
        idx = np.random.choice(
            data.shape[0], size=self.n_cluster, replace=False)
        self.centroids = np.copy(data[idx, :])

    def fit(self, data):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix(M data points with D attributes each)(numpy float)
        Returns:
            Distance between old set of clusters and the new ones.
        """
        if data.shape[0] < self.n_cluster:
            raise ValueError(
                'Number of clusters is greater than number of datapoints')

        best_centroids = None
        m_score = float('inf')
        clusterDifference=0

        if(type(self.centroids) is not np.ndarray):
            if(self.centroids==None):
                self.init_centroids(data)

        old_centroid = np.copy(self.centroids)
        for _ in range(self.max_iter):
            cluster_assign = self.e_step(data)
            self.m_step(data, cluster_assign)

            if np.abs(old_centroid - self.centroids).sum() < self.delta:
                return np.abs(old_centroid - self.centroids).sum()
            clusterDifference = np.abs(old_centroid - self.centroids).sum()
        cur_score = self.evaluate(data)

        if cur_score < m_score:
            m_score = cur_score
            best_centroids = np.copy(self.centroids)

        self.centroids = best_centroids

        return clusterDifference

    def e_step(self, data):
        """
        Expectation Step.
        Finding the cluster assignments of all the points in the data passed
        based on the current centroids
        Args:
            data: M x D Matrix (M training samples with D attributes each)(numpy float)
        Returns:
            Cluster assignment of all the samples in the training data
            (M) Vector (M number of samples in the train dataset)(numpy int)
        """
        #TODO
        clusterList=[]
        for v in data:
            distanceList=[]
            for index in range(self.n_cluster):
                distanceList.append(np.square(np.linalg.norm(v-self.centroids[index])))
            clusterList.append(argmin(distanceList))
        return np.array(clusterList,dtype=int)


    def m_step(self, data, cluster_assgn):
        """
        Maximization Step.
        Compute the centroids
        Args:
            data: M x D Matrix(M training samples with D attributes each)(numpy float)
        Change self.centroids
        """
        #TODO
        for index in range(self.n_cluster):
            elementsList=[]
            for i in range(len(cluster_assgn)):
                if(cluster_assgn[i]==index):
                    elementsList.append(data[i])
            if(len(elementsList)!=0):
                self.centroids[index]=np.mean(elementsList,axis=0)

    def evaluate(self, data):
        """
        K-Means Objective
        Args:
            data: Test data (M x D) matrix (numpy float)
        Returns:
            metric : (float.)
        """
        #TODO
        totalerror=0
        cluster_assign=self.e_step(data)
        for index in range(self.n_cluster):
            distanceList=[]
            for i in range(data.shape[0]):
                if(cluster_assign[i]==index):
                    distanceList.append(np.square(np.linalg.norm(data[i,:]-self.centroids[index])))
            totalerror+=np.sum(distanceList)
        return totalerror

    def predict(self,data):
        """
            Args:
                data: Test data(M X D) matrix (numpy float)
            Returns:
                cluster assignment: Clusters to which the input belongs to. 
        """
        return self.e_step(data) 
