#importing spark related libraries
from pyspark.sql.functions import *

#importing other necessary libraries 
import numpy as np
import os

from sklearn.decomposition import TruncatedSVD

#importing necessary files
from dataExploration import dataExploration
from evaluate import evaluate
from model import model
from preprocess import preprocess
from cluster import cluster

def readStream(rdd,ssc,spark,spark_context,classifierModel,clusteringModel,modelChosen,parameters,testingParams,emptyRDD_count=None):
  schema=parameters["schema"]
  op=parameters["op"]
  isClustering=parameters["isClustering"]
  endless=parameters["endless"]
  explore=parameters["explore"]
  proc=parameters["proc"]
  sampleFrac=parameters["sf"]
  hashmap_size=parameters["hashmap_size"]


  if not rdd.isEmpty():
    emptyRDD_count[0]=0

    print('\nStarted Processing a Batch')
    print("----------------------------")
    #parsing json to get a df
    print("-> Started Parsing JSON")
    df = spark.read.json(rdd)
    df.cache()

    # applying schema to it
    newdf=spark.createDataFrame(data=spark_context.emptyRDD(),schema=schema)
    n_samples = len(df.columns)
    for rowNumber in range(n_samples):
      newdf=newdf.union(df.withColumn(str(rowNumber),to_json(col(str(rowNumber))))\
        .select(json_tuple(col(str(rowNumber)),"feature0","feature1","feature2"))\
          .toDF("Subject","Body","Spam/Ham"))
    df.unpersist()
    
    #sampling the df because not everything can be stored in memory 
    newdf=newdf.sample(fraction=sampleFrac,seed=10)
    
    lengthdf=dataExploration(newdf,explore)
    clean_df=preprocess(lengthdf,hashmap_size,proc)
    if(isClustering==False):
      if(op=="train"):
        model(clean_df,classifierModel)
        # uncomment only when training accuracy is required
        # X=np.array(clean_df.select('features').collect())
        # y=np.array(clean_df.select('label').collect())
        # #training accuracy
        # predictions=classifierModel.predict(X.reshape(X.shape[0],X.shape[2]))
        # evaluate(predictions,y.reshape(y.shape[0]),testingParams,op)

      elif(op=="test"):
        clean_df.cache()
        X=np.array(clean_df.select('features').collect())
        y=np.array(clean_df.select('label').collect())
        clean_df.unpersist()
        predictions=classifierModel.predict(X.reshape(X.shape[0],X.shape[2]))
        evaluate(predictions,y.reshape(y.shape[0]),testingParams,op)

    else:#cluster
      if(op=="train"):
        cluster(clean_df,clusteringModel,ssc,modelChosen,endless)
      else:
        X = np.array(clean_df.select('features').collect())
        X = X.reshape(X.shape[0],X.shape[2])
        y=np.array(clean_df.select('label').collect())

        tsvd = TruncatedSVD(n_components=10)
        data = tsvd.fit_transform(X[:,:-1])
        class_pred = clusteringModel.predict(data)
        class_pred = np.reshape(class_pred,(class_pred.shape[0],1))

        data = np.concatenate((data, class_pred,y), axis=1)
        X = class_pred = None

        print("-> Saving points and their class to a file")
        arr = np.array([])
        if(os.path.isfile("clusteringModels/results.npy")):
          f = open('clusteringModels/results.npy','rb')
          arr = np.load(f,allow_pickle=True)
          f.close()
        f = open('clusteringModels/results.npy','wb')
        if(arr.shape[0]!=0):
          print("Current batch data shape:",data.shape)
          print(f"Previously stored data shape{arr.shape}")
          data = np.concatenate((arr,data))
          print("Concatentated Data shape:",data.shape)
        arr= None
        np.save(f,data)
        f.close()

  else:#rdd is empty
    emptyRDD_count[0]+=1
    if(emptyRDD_count[0]==3):#if 3 empty rdds are received, assume streaming has stopped
      ssc.stop()