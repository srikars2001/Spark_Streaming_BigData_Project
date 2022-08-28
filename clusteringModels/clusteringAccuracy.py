import numpy as np

#f = open('clusteringModels/results.npy','rb')
f = open('results.npy','rb')
arr = np.load(f, allow_pickle=True)
f.close()

cluster_pred=arr[:,-2]
actual_classes=arr[:,-1]
correct_classification = np.sum(cluster_pred==actual_classes)
accuracy = correct_classification/np.shape(arr)[0]

print("Accuracy For comparing Clusters with actual Classes:",accuracy)

indices_zero = cluster_pred==0
indices_one = cluster_pred==1
cluster_pred[indices_zero]=1
cluster_pred[indices_one]=0

correct_classification = np.sum(cluster_pred==actual_classes)
accuracy = correct_classification/np.shape(arr)[0]

print("Accuracy For comparing Clusters with actual Classes:",accuracy)