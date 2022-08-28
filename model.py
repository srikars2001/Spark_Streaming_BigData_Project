from pyspark.sql.functions import *
import numpy as np

from sklearn.utils import parallel_backend


def model(df,classifierModel):
    print("-> Entered Model Building Stage")

    #converting df to numpy array
    X=np.array(df.select('features').collect())
    y=np.array(df.select('label').collect())

    #reshaping
    print("Shape of X:",X.shape)
    X=X.reshape(X.shape[0],X.shape[2])
    print("Shape of y:",y.shape)
    y=y.reshape(y.shape[0])

    #with parallel_backend('spark',n_jobs=-1):
    classifierModel.partial_fit(X,y,classes=list(range(2)))
