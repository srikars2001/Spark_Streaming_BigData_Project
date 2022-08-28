#importing spark related libraries
from pyspark import SparkContext
from pyspark.sql.functions import *
from pyspark.sql import SQLContext,SparkSession
from pyspark.streaming import StreamingContext
from pyspark.sql.types import *
from pyspark.sql.functions import *

#importing other necessary files
from preprocess import preprocess
from helper import *

#import different model libraries
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.cluster import MiniBatchKMeans, Birch
from customKmeans.KMeans import KMeansClustering

#importing other necessary libraries 
import argparse
import pickle

from readStream import readStream

# joblib libraries for parallel processing of sklearn models-> uncomment if it improves performance
# from joblibspark import register_spark
# register_spark()#register joblib with spark backend

# Run using /opt/spark/bin/spark-submit main.py -host <hostname> -p <port_no> -b <batch_size> -t <isTest> -m <model_name>
parser = argparse.ArgumentParser(
    description="main driver program which calls rest of the files")
addArguments(parser)

 
if __name__ == '__main__':
  #get command line argumetns
  args = parser.parse_args()
  print(args)

  #declaration of command line argument values
  hostname=args.host_name
  port=args.port_number
  window_interval=args.window_interval
  op=args.op
  proc=args.proc
  sf=args.sampleFraction
  modelChosen=args.model
  isClustering=args.cluster
  explore=args.explore
  hashmap_size=args.hashmap_size
  endless=args.endless

  #storing initial data for visualization purposes
  if(explore==True):
    spam_count_viz=[0]
    ham_count_viz=[0]
    with open('./visualizations/spam.pkl','wb') as f:
      pickle.dump(spam_count_viz,f)
      pickle.dump(ham_count_viz,f)

  #initialisation of spark context and streaming spark context
  spark_context = SparkContext.getOrCreate()
  spark=SparkSession(spark_context)
  ssc=StreamingContext(spark_context,window_interval)

  stream_data=ssc.socketTextStream(hostname,int(port))

  schema = StructType(
      [StructField("Subject",StringType(),True),
      StructField("Body",StringType(),True),
      StructField("Spam/Ham",StringType(),True)])

  classifierModel, clusteringModel = initializeModel(op, isClustering, modelChosen, endless,proc) 

  emptyRDD_count=[0]# for keeping track of empty rdds
  testingParams={'tp':0,'tn':0,'fp':0,'fn':0}# for keeping track of test metrics
  parameters = {
    "schema":schema,"endless":endless,
    "op":op,"proc":proc,"sf":sf,
    "hashmap_size":hashmap_size,
    "isClustering":isClustering,
    "explore" : explore
  }
  stream_data.foreachRDD(lambda rdd:readStream(rdd,ssc,\
    spark,spark_context,classifierModel,clusteringModel,\
      modelChosen,parameters,testingParams,emptyRDD_count))

  ssc.start()
  ssc.awaitTermination()
  
  if(op=="train"):
    if(isClustering==False):
      pickle.dump(classifierModel,open(f'trainedClassifierModels/with{proc}/{modelChosen}','wb'))
    else:#cluster
      print("Pickling cluster model")
      pickle.dump(clusteringModel,open(f'clusteringModels/{modelChosen}','wb'))
  elif(op=="test"):
    if(isClustering==False):
      #Print test metrics
      printMetrics(testingParams,modelChosen)
    else:#cluster
      #plot clusters
      plotClusters()
