import matplotlib.pyplot as plt
import numpy as np


f = open('clusterDifferences.npy','rb')
arr = np.load(f, allow_pickle=True)
f.close()

#generating batch numbers list
batches=list(range(1,len(arr)+1))

plt.plot(batches,arr)
plt.xlabel('Batch Number')
plt.ylabel('Difference in Cluster Distance')
plt.title('Cluster Difference for an endless stream in Custom KMeans')
plt.show()