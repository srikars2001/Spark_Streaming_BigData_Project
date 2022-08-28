from pyspark import SparkContext
from pyspark.sql.functions import *
from pyspark.sql import SQLContext,SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
import pickle

def dataExploration(df,explore):
    if(explore==True):
        print("-> Started Data Exploration")
        ham_count=df.filter(df["Spam/Ham"]=="ham").count()
        spam_count=df.filter(df["Spam/Ham"]=="spam").count()
        #inserting into file for visualization purposes
        with open('./visualizations/spam.pkl','rb') as f1:
            spam_count_viz=pickle.load(f1)
            ham_count_viz=pickle.load(f1)
            """ print(type(spam_count_viz))
            print(type(ham_count_viz)) """
        #updating values 
        """ print(spam_count_viz) """   
        spam_count_viz.append(spam_count)
        ham_count_viz.append(ham_count)
        """ print("spam_count_viz",spam_count_viz)
        print("ham_count_viz",ham_count_viz) """
        #dumping
        with open('./visualizations/spam.pkl','wb') as f:
            pickle.dump(spam_count_viz,f)
            pickle.dump(ham_count_viz,f)

        """ with open('./visualizations/spam.pkl','rb') as f1:
            spam_count_viz=pickle.load(f1)
            ham_count_viz=pickle.load(f1)
        print("after spam_count_viz ",spam_count_viz)
        print("after ham_count_viz ",ham_count_viz) """   
        total_count=ham_count+spam_count
        ham_percent=ham_count/total_count
        spam_percent=spam_count/total_count
        print("Ham messages:",ham_count," (",ham_percent*100,"%)")
        print("Spam messages:",spam_count," (",spam_percent*100,"%)")

    length_df = df.withColumn('length',length(df['Body']))

    if(explore==True):
        length_df.show(3)

        meandf=length_df.groupBy('Spam/Ham').mean()
        vardf=length_df.groupBy('Spam/Ham').agg({'length':'variance'})
        vardf=vardf.withColumnRenamed("Spam/Ham","s/h")
        uniondf=meandf.join(vardf,meandf['Spam/Ham']==vardf['s/h'],how="inner").drop('s/h').show()

    return length_df