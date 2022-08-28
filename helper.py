#All helper functions are defined here

#import different model libraries
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.cluster import MiniBatchKMeans, Birch
from customKmeans.KMeans import KMeansClustering

import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

def addArguments(parser):
    """Adds command line arguments to parser object.
    Input; parser(expects a parser argument)
    Output: Adds various necessary command line arguments
    """
    parser.add_argument('--host-name', '-host', help='Hostname', required=False,
                    type=str, default="localhost") 
    parser.add_argument('--port-number', '-p', help='Port Number', required=False,
                        type=int, default=6100) 
    parser.add_argument('--window_interval', '-w', help='Window Interval', required=False,
                        type=int, default=5) 
    parser.add_argument('--op', '-op', help='Operation being performed', required=False,
                        type=str, default="train") # op can be 1 among 'train','test'
    parser.add_argument('--proc', '-proc', help='Type of Preprocessing Performed', required=False,
                        type=str, default="tf")# choose 1 among tf,word2vec,glove,use,elmo or bert
    parser.add_argument('--sampleFraction','-sf',help='Sampling fraction for every batch',required=False,
                        type=float,default=1.0) # Use this when each batch size is large
    parser.add_argument('--model', '-m', help='Choose Model', required=False,
                        type=str, default="NB")#model can be 1 among 'NB','SVM','LR','MLP','PA','KMeans' or 'Birch'
    parser.add_argument('--cluster', '-c', help='Enable clustering',
                        required=False, type=bool, default=False)
    parser.add_argument('--endless', '-endless', help='Streaming is Endless',
                        required=False, type=bool, default=False)
    parser.add_argument('--explore', '-e', help='Enable data exploration',
                        required=False, type=bool, default=False)
    parser.add_argument('--hashmap_size', '-hash', help='Hash map size to be used', required=False,
                        type=int, default=14)#hashmap_size=2^(this number)

def initializeModel(op,isClustering,modelChosen,endless,proc):
    """
    Helps in choosing the classifier or the clustering based on the 
    command line arguments sent as parameters to this function.
    """
    classifierModel=None
    clusteringModel=None
    if(op=="train"):
        if(isClustering==False):
            if(modelChosen=="NB"):
                classifierModel = MultinomialNB()
            elif(modelChosen=="SVM"):
                classifierModel = SGDClassifier(alpha=0.0001,learning_rate='adaptive',eta0=0.5,n_jobs=-1,n_iter_no_change=1000)
            elif(modelChosen=="LR"):
                classifierModel = SGDClassifier(loss="log")
            elif(modelChosen=="MLP"):
                classifierModel = MLPClassifier(activation="logistic")
            else:
                classifierModel = PassiveAggressiveClassifier(n_jobs=-1,C=0.5,random_state=5)
        else:
            if(endless==True):
                clusteringModel = KMeansClustering(n_clusters=2)
            else:
                if(modelChosen=="KMeans"):
                    clusteringModel = MiniBatchKMeans(n_clusters=2, random_state=123)
                else:#Birch
                    clusteringModel = Birch(n_clusters=2)
    elif(op=="test"):
        if(isClustering==False):
            classifierModel = pickle.load(open(f'trainedClassifierModels/with{proc}/{modelChosen}', 'rb'))
        else:#cluster
            clusteringModel = pickle.load(open(f'clusteringModels/{modelChosen}', 'rb'))
    return (classifierModel,clusteringModel)


def printMetrics(testingParams,modelChosen):
      total_samples=testingParams['tp']+testingParams['tn']+testingParams['fp']+testingParams['fn']
      accuracy=(testingParams['tp']+testingParams['tn'])/total_samples
      precision=(testingParams['tp'])/(testingParams['tp']+testingParams['fp'])
      recall=(testingParams['tp'])/(testingParams['tp']+testingParams['fn'])
      f1=(2*precision*recall)/(precision+recall)

      print(f"Model Name: {modelChosen}")
      print("----------------------------------")
      print("")
      print("Confusion Matrix:")
      print("---------------")
      print(f"{testingParams['tp']} | {testingParams['fn']}")
      print("---------------")
      print(f"{testingParams['fp']} | {testingParams['tn']}")
      print("---------------")
      print("")
      print("Accuracy: {:.4f}".format(accuracy))
      print("Precision: {:.4f}".format(precision))
      print("Recall: {:.4f}".format(recall))
      print("F1 Score: {:.4f}".format(f1))

def plotClusters():
    """Plots clusters for visualization"""
    f = open('clusteringModels/results.npy','rb')
    arr = np.load(f, allow_pickle=True)
    f.close()

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(arr[:,:-2])
    print('Explained variance by the selected principal component: {}'.format(np.sum(pca.explained_variance_ratio_)))

    fig=plt.figure()
    ax = fig.add_subplot(121,projection="3d")
    ax.scatter(
        xs=pca_result[:,0], 
        ys=pca_result[:,1], 
        zs=pca_result[:,2], 
        c=arr[:,-2], 
        linewidths=3,
        cmap='tab10'
    )
    ax.set_title("Cluster Visualization in 3D")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")

    ax = fig.add_subplot(122)
    ax.scatter(
        x=pca_result[:,0], 
        y=pca_result[:,1], 
        c=arr[:,-1], 
        linewidths=3,
        cmap='tab10',
    )
    ax.set_title("Cluster Visualization in 2D")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    plt.show()