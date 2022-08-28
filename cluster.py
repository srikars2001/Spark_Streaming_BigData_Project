from pyspark.sql.functions import *
import numpy as np
import os

from sklearn.utils import parallel_backend
from sklearn.decomposition import TruncatedSVD

def cluster(df,clusteringModel,ssc, chosenModel,endless):
    print("-> Entered Cluster Model Building Stage")

    #reshaping
    X=np.array(df.select('features').collect())
    print("Shape of X:",X.shape)
    X=np.reshape(X,(X.shape[0],X.shape[2]))

    tsvd = TruncatedSVD(n_components=10)
    tsvd_result = tsvd.fit_transform(X[:,:-1])
    print(f"Cumulative variance explained by the selected components: {np.sum(tsvd.explained_variance_ratio_)}")
    if(endless==True or chosenModel=='CustomKMeans'):
        clusterDifference = clusteringModel.fit(tsvd_result)
        print("Difference between clusters:",clusterDifference)
        arr = np.array([])
        data = np.array([clusterDifference])
        if(os.path.isfile("clusteringModels/clusterDifferences.npy")):
          f = open('clusteringModels/clusterDifferences.npy','rb')
          arr = np.load(f,allow_pickle=True)
          f.close()
        f = open('clusteringModels/clusterDifferences.npy','wb')
        if(arr.shape[0]!=0):
          data = np.concatenate((arr,data))
        arr= None
        np.save(f,data)
        f.close()
        if(clusterDifference<0.001):
            ssc.stop()
    else:
        #with parallel_backend('spark',n_jobs=-1):
        clusteringModel.partial_fit(tsvd_result)
        if(chosenModel=='KMeans'):
            score = -1 * clusteringModel.score(tsvd_result)
            print("Score:",score)