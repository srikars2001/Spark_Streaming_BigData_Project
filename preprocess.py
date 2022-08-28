from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer,HashingTF, Tokenizer, StopWordsRemover ,\
    VectorAssembler, IDF, StringIndexer,StandardScaler, Word2Vec, RegexTokenizer

#importing spark-nlp related libraries
# import sparknlp
# sparknlp.start()
# #sparknlp.start(gpu=True) #for training on GPU
# from sparknlp.base import *
# from sparknlp.annotator import *

# def USE_Embedding(df,procMethod):
#     document_assembler = DocumentAssembler().setInputCol("data").setOutputCol("document") \
#         .setCleanupMode("shrink")
#     tokenizer = Tokenizer() \
#         .setInputCols(["document"]) \
#         .setOutputCol("token") \
#         .setSplitChars(['-']) \
#         .setContextChars(['(', ')', '?', '!', '#', '@']) 
#     normalizer = Normalizer() \
#         .setInputCols(["token"]) \
#         .setOutputCol("normalized")\
#         .setCleanupPatterns(["[^\w\d\s]"])#to remove punctuations
#     stopwords_cleaner = StopWordsCleaner()\
#         .setInputCols("normalized")\
#         .setOutputCol("cleanTokens")\
#         .setCaseSensitive(False) 
#     lemma = LemmatizerModel.pretrained('lemma_antbnc') \
#         .setInputCols(["cleanTokens"]) \
#         .setOutputCol("lemma")
    
#     embeddingModel = None
#     if(procMethod=='USE'):
#         embeddingModel = UniversalSentenceEncoder.pretrained().setInputCols(["document","lemma"]).setOutputCol("embeddings")
#     elif(procMethod=='glove'):
#         embeddingModel = WordEmbeddingsModel().pretrained() \
#             .setInputCols(["document",'lemma'])\
#             .setOutputCol("embeddings")\
#             .setCaseSensitive(False)
#     elif(procMethod=='elmo'):
#         embeddingModel = ElmoEmbeddings.pretrained('elmo')\
#           .setInputCols(["document", "token"])\
#           .setOutputCol("embeddings")
#     else:#bert
#         embeddingModel = BertEmbeddings().pretrained(name='bert_base_cased', lang='en') \
#             .setInputCols(["document",'token'])\
#             .setOutputCol("embeddings")
    
#     embeddingsSentence = SentenceEmbeddings() \
#         .setInputCols(["document", "embeddings"]) \
#         .setOutputCol("features") \
#         .setPoolingStrategy("AVERAGE")
#     pipeline = Pipeline(stages = [document_assembler, tokenizer, normalizer, lemma, embeddingModel, embeddingsSentence])
#     cleaned_df = pipeline.fit(df)
#     clean_df = cleaned_df.transform(df)
#     print(clean_df.show())

#     return clean_df


def preprocess(df,hashmap_size,proc):
    print("-> Entered Preprocessing Stage")

    #concatenating both body and subject into a single column
    df = df.withColumn('data',concat(col('Subject'),lit(" "),col("Body")))
    df = df.drop('Subject','Body')
    #call spark-nlp models here

    #Feature extraction
    # df=df.select(regexp_replace(col('data'),'\\p{Punct}','').alias('data'),'length','Spam/Ham')#removing punctuations
    # tokenizer = Tokenizer(inputCol = 'data', outputCol = 'tokens')
    ham_spam_to_numeric = StringIndexer(inputCol = 'Spam/Ham', outputCol = 'label',stringOrderType ='alphabetAsc')
    
    clean_df = None
    pipeline = None
    if(proc=='word2vec'):
        tokenizer = Tokenizer(inputCol = 'data', outputCol = 'tokens')
        word2vec = Word2Vec(vectorSize=300, seed=12, inputCol='tokens', outputCol='features')
        pipeline = Pipeline(stages=[ham_spam_to_numeric, tokenizer, word2vec])
    else:#tf
        regexTokenizer = RegexTokenizer(inputCol =  'data', outputCol = 'tokens', pattern="[\\p{Punct}\\s]+")
        stop_remove = StopWordsRemover(inputCol = 'tokens', outputCol = 'stop_token')
        #count_vec = CountVectorizer(inputCol = 'stop_token', outputCol = 'c_vec')
        hashmap =  HashingTF(inputCol='stop_token', outputCol = 'h_vec',numFeatures=2**hashmap_size)#16384,32768
        idf = IDF(inputCol = 'h_vec', outputCol = 'tf_idf')

        clean_up = VectorAssembler(inputCols = ['tf_idf', 'length'], outputCol = 'features')

        pipeline = Pipeline(stages=[ham_spam_to_numeric, regexTokenizer, stop_remove, hashmap, idf, clean_up])
    pipelineFit = pipeline.fit(df)
    clean_df = pipelineFit.transform(df)
    clean_df = clean_df.select('features','label')
    #clean_df.show(3)
    return clean_df