// Databricks notebook source
// DBTITLE 1,Read in CSV files
val df = spark.table("train_csv")
val test_df = spark.table("test_csv")
display(df)

// COMMAND ----------

// DBTITLE 1,Tokenize ingredients & Remove non-alphabetical characters
// convert ingredient list from String to Seq (all lowercase)
import org.apache.spark.sql.functions._

val makeSeq : String => Array[String] = _.toLowerCase().replaceAll("-"," ").replaceAll("[^a-zA-Z ]+","").replaceAll(" +"," ").split(" ").filterNot(x=>x=="").filterNot(x=>x=="oz")
val makeSeqUDF = udf(makeSeq)
val df1 = df.withColumn("ingredients_list", makeSeqUDF('ingredients))
val test_df1 = test_df.withColumn("ingredients_list", makeSeqUDF('ingredients))
display(df1)

// COMMAND ----------

// DBTITLE 1,Check # of Unique Tokens
// find the total number of unique ingredients
val all_ingrs = df1.select(explode(col("ingredients_list"))).withColumnRenamed("col", "ingredient")
val unique_ingrs = all_ingrs.distinct()
unique_ingrs.count()

// COMMAND ----------

// DBTITLE 1,Plot Class Frequency
val cuisines = df1.select("cuisine")
val cuisines_count = cuisines.groupBy("cuisine").count().orderBy(desc("count"))
display(cuisines_count)
// unbalanced dataset (may need to stratify labels for train/validation/test set)
// e.g. there are almost 20 times more italian cuisines than brazilian ones

// COMMAND ----------

// DBTITLE 1,EDA : Find Top (most frequent) Tokens of a Cuisine
// Get top N unique ingredients for each cuisine (output: 20 sets)
def TopTokens(df: org.apache.spark.sql.DataFrame, cuisine: String, n_ingrs: Int) : org.apache.spark.sql.DataFrame = { //org.apache.spark.sql.DataFrame = {
  val top_tokens = df.filter(df("cuisine")===cuisine).select(explode(col("ingredients_list"))).withColumnRenamed("col", "token").select("token")
                     .groupBy("token").count().orderBy(desc("count")).withColumn("id", monotonically_increasing_id).where($"id"<n_ingrs)
  return top_tokens
}

// COMMAND ----------

// DBTITLE 1,Italian: Top 10 
display(TopTokens(df1,"italian",10))

// COMMAND ----------

// DBTITLE 1,Mexican: Top 10 
display(TopTokens(df1,"mexican",10))

// COMMAND ----------

// DBTITLE 1,Indian : Top 10 
display(TopTokens(df1,"indian",10))

// COMMAND ----------

// DBTITLE 1,Vietnamese : Top 10 
display(TopTokens(df1,"vietnamese",10))

// COMMAND ----------

// DBTITLE 1,Korean : Top 10 
display(TopTokens(df1,"korean",10))

// COMMAND ----------

// DBTITLE 1,Greek : Top 10 
display(TopTokens(df1,"greek",10))

// COMMAND ----------

// DBTITLE 1,Thai : Top 10 
display(TopTokens(df1,"thai",10))

// COMMAND ----------

// DBTITLE 1,Japanese : Top 10 
display(TopTokens(df1,"japanese",10))

// COMMAND ----------

// DBTITLE 1,Chinese : Top 10 
display(TopTokens(df1,"chinese",10))

// COMMAND ----------

// DBTITLE 1,Perform STRATIFIED train-test split
import org.apache.spark.sql.{DataFrame, DataFrameStatFunctions}  // Library which includes sampleBy

// This function will produce a "stratified" train-test split, which can be useful when dealing with an imbalanced dataset such as this one
def train_test_split_stratified(test_fraction: Double, input_dataframe: DataFrame, stratified: Boolean): (DataFrame,DataFrame) = {
  if(stratified == true){  
    val column_names = input_dataframe.columns.toSeq   // Extract list of column names from input dataframe
    val cuisine_array = input_dataframe.select("cuisine").distinct.orderBy(asc("cuisine")).collect().map(array_element => array_element(0))   // Extract list of cuisines
    val cuisine_list = cuisine_array.toList   
    val training_fraction_list = List.fill(cuisine_list.length)(1 - test_fraction)   // List of training fractions for each class (if stratified: SAME for all classes)
    // Takes the list of cuisines and maps each of them to the required training fraction
    // This mapping is required as an input of the sampleBy method
    val training_factors = (cuisine_list zip training_fraction_list).toMap
    val training_set = input_dataframe.stat.sampleBy("cuisine", training_factors, 7L)   // Perform stratified split on the original dataset & store training set
    // Now, the rows from the original dataset which are not found in the training set will be allocated to the test set
    val training_set_join =  training_set.select(training_set.columns.map { c => training_set.col(c).as( c + "_1") } : _* )
    // Perform an outer join between the original dataframe and the newly created dataset
    // All rows excluding those of the training set will be set to those of the test sets in the two commands below
    val df_training_match = input_dataframe.join(training_set_join, input_dataframe.col("_c0") === training_set_join.col("_c0_1"),"outer")
    val test_set = df_training_match.filter("_c0_1 is null").select(column_names.map(c => col(c)): _*)
    return (training_set,test_set)   
  } 
  else {   // Return randomly split training and test sets
    val Array(training_set_random, test_set_random) = input_dataframe.randomSplit(Array(1 - test_fraction, test_fraction), 7L)
    return (training_set_random,test_set_random)      
  }
}

// COMMAND ----------

val (df_train, df_valid) = train_test_split_stratified(0.2, df1, true)  // stratified split
display(df_valid)

// COMMAND ----------

// # of data in training & valid sets
print(df_train.count(), df_valid.count()) 

// COMMAND ----------

// DBTITLE 1,Remove Duplicates Ingredients List from Training Data
val df_train_NOduplicate = df_train.withColumn("sorted_ingredients_list", sort_array($"ingredients_list")).dropDuplicates("sorted_ingredients_list")
df_train_NOduplicate.count()   //about 350 duplicates gone!

// COMMAND ----------

// DBTITLE 1,Assign Class Weights (for class imbalance)
import org.apache.spark.sql.expressions.Window 

val n_classes = 20
val partitionWindow = Window.partitionBy($"cuisine")
// count # of training data & # of data for EACH class (class size)
val df_train_classcount = df_train_NOduplicate.withColumn("total_nRecords", lit(df_train.count())).select($"*", count('cuisine) over partitionWindow as "cuisine_count")
// assign weights inversely proportional to class size
val df_train_classweights = df_train_classcount.withColumn("cuisine_weight", ($"total_nRecords"/$"cuisine_count")/n_classes) 
// display assigned weight for each class (from smallest to biggest)
display(df_train_classweights.select("cuisine", "cuisine_weight").distinct().orderBy("cuisine_weight")) 

// COMMAND ----------

// DBTITLE 1,Cross Validation
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.{StringIndexer, CountVectorizer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val indexer = new StringIndexer().setInputCol("cuisine").setOutputCol("label")
val countV = new CountVectorizer().setInputCol("ingredients_list").setOutputCol("features")
val rf = new RandomForestClassifier().setNumTrees(100).setWeightCol("cuisine_weight")
val pipeline = new Pipeline().setStages(Array(indexer, countV, rf))

// We use a ParamGridBuilder to construct a grid of parameters to search over.
// With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
// this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
val paramGrid = new ParamGridBuilder()
  .addGrid(rf.maxDepth, Array(26, 28, 30))
  .build()

// We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
// This will allow us to jointly choose parameters for all Pipeline stages.
// A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
// It uses MulticlassClassificationEvaluator and weighted f1-score metric.
val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(new MulticlassClassificationEvaluator().setMetricName("weightedFMeasure"))
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3) 
  .setParallelism(3)  // Evaluate up to 2 parameter settings in parallel

// Run cross-validation, and choose the best set of parameters.
val cvModel = cv.fit(df_train_classweights)

// COMMAND ----------

// DBTITLE 1,Function to Compute Macro-averaged F1 Score
import org.apache.spark.mllib.evaluation.MulticlassMetrics
def F1_macro_average(metrics_input: org.apache.spark.mllib.evaluation.MulticlassMetrics): Double = {  
  val labels = metrics_input.labels   // From multiclass metrics object extract labels into a list 
  var F1_sum: Double = 0   // Initialize F1_sum value as 0
  // for loop : Add each label's F1 score together
  labels.foreach { 
    l => F1_sum = metrics_input.fMeasure(l) + F1_sum 
    println(l, metrics_input.fMeasure(l))
  }  
  return(F1_sum/metrics_input.labels.length)   // Return the macro-averaged F1 (sum of F1-scores divided by number of classes)
}

// COMMAND ----------

// DBTITLE 1,Template : (CountVectorizer + Random Forest) 
// (Random Forest + countVectorizer) Trial 1 : ALL ingredients, maxDepth=5, minInstancesPerNode=1
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.{StringIndexer, CountVectorizer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val indexer = new StringIndexer().setInputCol("cuisine").setOutputCol("cuisine_idx")
val countV = new CountVectorizer().setInputCol("ingredients_list").setOutputCol("ingredients_countV")
val rf = new RandomForestClassifier().setNumTrees(100).setMaxDepth(30).setFeaturesCol("ingredients_countV").setLabelCol("cuisine_idx").setWeightCol("cuisine_weight")

val pipeline = new Pipeline().setStages(Array(indexer, countV, rf))
val rf30 = pipeline.fit(df_train_classweights)  

// COMMAND ----------

// DBTITLE 1,CountVectorizer  + Random Forest : maxDepth = 18
// training took 4 minutes
val train_preds = rf18.transform(df_train)
val predictionsAndLabels_train = train_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics = new MulticlassMetrics(predictionsAndLabels_train)   // Instantiate metrics object
val macro_F1 = F1_macro_average(metrics)

val val_preds = rf18.transform(df_valid)
val predictionsAndLabels_val = val_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics_val = new MulticlassMetrics(predictionsAndLabels_val)   // Instantiate metrics object
val macro_F1_val = F1_macro_average(metrics_val)

// COMMAND ----------

// DBTITLE 1,CountVectorizer + Random Forest : maxDepth = 20
// training took 8 minutes
val train_preds = rf20.transform(df_train)
val predictionsAndLabels_train = train_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics = new MulticlassMetrics(predictionsAndLabels_train)   // Instantiate metrics object
val macro_F1 = F1_macro_average(metrics)

val val_preds = rf20.transform(df_valid)
val predictionsAndLabels_val = val_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics_val = new MulticlassMetrics(predictionsAndLabels_val)   // Instantiate metrics object
val macro_F1_val = F1_macro_average(metrics_val)

// COMMAND ----------

// DBTITLE 1,CountVectorizer + Random Forest : maxDepth = 22
// training took 11 minutes
val train_preds = rf22.transform(df_train)
val predictionsAndLabels_train = train_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics = new MulticlassMetrics(predictionsAndLabels_train)   // Instantiate metrics object
val macro_F1 = F1_macro_average(metrics)

val val_preds = rf22.transform(df_valid)
val predictionsAndLabels_val = val_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics_val = new MulticlassMetrics(predictionsAndLabels_val)   // Instantiate metrics object
val macro_F1_val = F1_macro_average(metrics_val)

// COMMAND ----------

// DBTITLE 1,CountVectorizer + Random Forest : maxDepth = 24
// training took 14 minutes
val train_preds = rf24.transform(df_train)
val predictionsAndLabels_train = train_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics = new MulticlassMetrics(predictionsAndLabels_train)   // Instantiate metrics object
val macro_F1 = F1_macro_average(metrics)

val val_preds = rf24.transform(df_valid)
val predictionsAndLabels_val = val_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics_val = new MulticlassMetrics(predictionsAndLabels_val)   // Instantiate metrics object
val macro_F1_val = F1_macro_average(metrics_val)

// COMMAND ----------

// DBTITLE 1,CountVectorizer + Random Forest : maxDepth = 26
// training took 18.5 minutes
val train_preds = rf26.transform(df_train)
val predictionsAndLabels_train = train_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics = new MulticlassMetrics(predictionsAndLabels_train)   // Instantiate metrics object
val macro_F1 = F1_macro_average(metrics)

val val_preds = rf26.transform(df_valid)
val predictionsAndLabels_val = val_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics_val = new MulticlassMetrics(predictionsAndLabels_val)   // Instantiate metrics object
val macro_F1_val = F1_macro_average(metrics_val)

// COMMAND ----------

// DBTITLE 1,CountVectorizer + Random Forest : maxDepth = 28
// training took 23.5 minutes
val train_preds = rf28.transform(df_train)
val predictionsAndLabels_train = train_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics = new MulticlassMetrics(predictionsAndLabels_train)   // Instantiate metrics object
val macro_F1 = F1_macro_average(metrics)

val val_preds = rf28.transform(df_valid)
val predictionsAndLabels_val = val_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics_val = new MulticlassMetrics(predictionsAndLabels_val)   // Instantiate metrics object
val macro_F1_val = F1_macro_average(metrics_val)

// COMMAND ----------

// DBTITLE 1,CountVectorizer + Random Forest : maxDepth = 30
// training took 30 minutes
val train_preds = rf30.transform(df_train)
val predictionsAndLabels_train = train_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics = new MulticlassMetrics(predictionsAndLabels_train)   // Instantiate metrics object
val macro_F1 = F1_macro_average(metrics)

val val_preds = rf30.transform(df_valid)
val predictionsAndLabels_val = val_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics_val = new MulticlassMetrics(predictionsAndLabels_val)   // Instantiate metrics object
val macro_F1_val = F1_macro_average(metrics_val)

// COMMAND ----------

// DBTITLE 1,Training Micro F1-Score
val train_preds = rf30.transform(df_train)
val predictionsAndLabels_train = train_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics = new MulticlassMetrics(predictionsAndLabels_train)   // Instantiate metrics object
val micro_F1_train = metrics.weightedFMeasure

// COMMAND ----------

// DBTITLE 1,Validation Micro F1-Score
val val_preds = rf30.transform(df_valid)
val predictionsAndLabels_val = val_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics_val = new MulticlassMetrics(predictionsAndLabels_val)   // Instantiate metrics object
val micro_F1_val = metrics_val.weightedFMeasure

// COMMAND ----------

// DBTITLE 1,Count Vectorizer Vocabulary
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}

// fit a CountVectorizerModel from the corpus
val cvModel: CountVectorizerModel = new CountVectorizer().setInputCol("ingredients_list").setOutputCol("ingredients_countV").fit(df_train)
cvModel.vocabulary

// COMMAND ----------

// DBTITLE 1,Compute Feature Importances
val importanceVector = rf30.stages(2).asInstanceOf[RandomForestClassificationModel].featureImportances
importanceVector.toArray.zipWithIndex.map(_.swap).sortBy(-_._2).foreach(x => println(cvModel.vocabulary(x._1) + " -> " + x._2))

// COMMAND ----------

// DBTITLE 1,Plot feature importance with the best model
val features = importanceVector.toArray.zipWithIndex.map(_.swap).sortBy(-_._2).map(x => cvModel.vocabulary(x._1)).toSeq.toDF("feature")
val featImportance_df = importanceVector.toArray.zipWithIndex.map(_.swap).sortBy(-_._2).toArray.toSeq.toDF("feature_idx","importance")
val idx2wordUDF  = udf((i:Int) => cvModel.vocabulary(i))
val featImportance_df1 = featImportance_df.withColumn("feature", idx2wordUDF($"feature_idx")).withColumn("index", monotonicallyIncreasingId)
display(featImportance_df1.where($"index"<10))

// COMMAND ----------

// SUM of all importances = 1
featImportance_df1.select("importance").rdd.map(_(0).asInstanceOf[Double]).reduce(_+_)

// COMMAND ----------

// DBTITLE 1,Confusion Matrix of the Best Model (CountVectorizer + maxDepth 30)
val val_preds = rf30.transform(df_valid)
val predictionsAndLabels_val = val_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics_val = new MulticlassMetrics(predictionsAndLabels_val)
val cols = (0 until metrics.confusionMatrix.numCols).toSeq
val cm = metrics_val.confusionMatrix.transpose.colIter.toSeq.map(_.toArray).toDF("arr")
val cm2 = cols.foldLeft(cm)((cm, i) => cm.withColumn("_" + (i+1), $"arr"(i))).drop("arr")
display(cm2)

// COMMAND ----------

// DBTITLE 1,Plot training & validation macro_f1 score vs. maxDepth
val f1_df = Seq((18, 0.7128673614685375, 0.6035635484503906), (20, 0.7418025922605114, 0.6134793074831031), (22, 0.7657907456341982, 0.6262298068210088),
                (24, 0.7875694586557167, 0.6310517879470019),  (26, 0.8181417190785878, 0.6484135297715679),  (28, 0.8357211888390651, 0.6541379126833322), 
                (30, 0.8523095808456332, 0.6630793763982988)).toDF("max depth", "train_macro_f1", "valid_macro_f1")
display(f1_df)

// COMMAND ----------

val test_preds = rf30.transform(test_df1)
display(test_preds.select("id", "prediction"))

// COMMAND ----------

// DBTITLE 1,Template : (tf-idf + Random Forest) 
import org.apache.spark.ml.feature.{StringIndexer, HashingTF, IDF}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.Pipeline

// Fit tf-idf feature & stringIndexer label embedding on train data 
val indexer = new StringIndexer().setInputCol("cuisine").setOutputCol("cuisine_idx")
val hashingTF = new HashingTF().setInputCol("ingredients_list").setOutputCol("ingredients_hashed").setNumFeatures(10000)
val idf = new IDF().setInputCol("ingredients_hashed").setOutputCol("ingredients_IDF")

// Fit Random Forest on train data 
val rf = new RandomForestClassifier().setNumTrees(100).setMaxDepth(28).setFeaturesCol("ingredients_IDF").setLabelCol("cuisine_idx").setWeightCol("cuisine_weight")
val pipeline = new Pipeline().setStages(Array(indexer, hashingTF, idf, rf))
val rf_tfidf28 = pipeline.fit(df_train_classweights) 

// COMMAND ----------

// DBTITLE 1,tf-idf + Random Forest : maxDepth = 28
// I will first check maxDepth = 28 and 30, since for CountVectorizer with sparse features, the best maxDepth was 30.
// So with TF-IDF, if performance with maxDepth of 30 is better than 28, we don't have to check lower values than 28.
// training took 15 minutes
val train_preds = rf_tfidf28.transform(df_train)
val predictionsAndLabels_train = train_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics = new MulticlassMetrics(predictionsAndLabels_train)   // Instantiate metrics object
val macro_F1 = F1_macro_average(metrics)
val val_preds = rf_tfidf28.transform(df_valid)
val predictionsAndLabels_val = val_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics_val = new MulticlassMetrics(predictionsAndLabels_val)   // Instantiate metrics object
val macro_F1_val = F1_macro_average(metrics_val)

// COMMAND ----------

// DBTITLE 1,tf-idf + Random Forest : maxDepth = 30
// training took 18 minutes
val train_preds = rf_tfidf30.transform(df_train)
val predictionsAndLabels_train = train_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics = new MulticlassMetrics(predictionsAndLabels_train)   // Instantiate metrics object
val macro_F1 = F1_macro_average(metrics)

val val_preds = rf_tfidf30.transform(df_valid)
val predictionsAndLabels_val = val_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics_val = new MulticlassMetrics(predictionsAndLabels_val)   // Instantiate metrics object
val macro_F1_val = F1_macro_average(metrics_val)

// COMMAND ----------

// <TF-IDF Results>
// maxDepth = 28 : (train_f1, valid_f1) = (0.767, 0.630)
// maxDepth = 30 : (train_f1, valid_f1) = (0.779, 0.636)
// performance with maxDepth of 30 is indeed better than 28 for both training & validation sets, so I don't have to check lower values than 28.
// maxDepth 30 is the best setting for TF-IDF.

// COMMAND ----------

// DBTITLE 1,Template : (Word2Vec + Random Forest) 
import org.apache.spark.ml.feature.{StringIndexer, Word2Vec, Word2VecModel}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}

// Fit Word2vec embedding on train data 
// # of records for the smallest class (brazilian) is 340. Should make the dimension less than this to prevent overfitting for less frequent labels.
val w2vModel: Word2VecModel = new Word2Vec().setInputCol("ingredients_list").setOutputCol("ingredients_w2v")
                                  .setVectorSize(128).setMinCount(1).setWindowSize(25).setMaxIter(15).fit(df_train) 
val df_train_w2v = w2vModel.transform(df_train_classweights)

// COMMAND ----------

// Fit StringIndexer on labels
val indexer = new StringIndexer().setInputCol("cuisine").setOutputCol("cuisine_idx").fit(df_train_w2v)
val df_train1 = indexer.transform(df_train_w2v)

// Fit Random Forest on w2v-transformed train data 
val rf = new RandomForestClassifier().setNumTrees(100).setMaxDepth(14).setFeaturesCol("ingredients_w2v").setLabelCol("cuisine_idx").setWeightCol("cuisine_weight")
val rf_w2v14= rf.fit(df_train1)  

// COMMAND ----------

// DBTITLE 1,word2vec + Random Forest : maxDepth = 10
// training took 4 minutes
val train_preds = rf_w2v10.transform(df_train1)
val predictionsAndLabels_train = train_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics = new MulticlassMetrics(predictionsAndLabels_train)   // Instantiate metrics object
val macro_F1 = F1_macro_average(metrics)

val df_val_w2v = w2vModel.transform(df_valid)                // fit word2vec trained on training data
val df_val1 = indexer.transform(df_val_w2v)
val val_preds = rf_w2v10.transform(df_val1)
val predictionsAndLabels_val = val_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics_val = new MulticlassMetrics(predictionsAndLabels_val)   // Instantiate metrics object
val macro_F1_val = F1_macro_average(metrics_val)

// COMMAND ----------

// DBTITLE 1,word2vec + Random Forest : maxDepth = 12
// training took 9 minutes
val train_preds = rf_w2v12.transform(df_train1)
val predictionsAndLabels_train = train_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics = new MulticlassMetrics(predictionsAndLabels_train)   // Instantiate metrics object
val macro_F1 = F1_macro_average(metrics)

val df_val_w2v = w2vModel.transform(df_valid)               // fit word2vec trained on training data
val df_val1 = indexer.transform(df_val_w2v)
val val_preds = rf_w2v12.transform(df_val1)
val predictionsAndLabels_val = val_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics_val = new MulticlassMetrics(predictionsAndLabels_val)   // Instantiate metrics object
val macro_F1_val = F1_macro_average(metrics_val)

// COMMAND ----------

// DBTITLE 1,word2vec + Random Forest : maxDepth = 14
// training took 17 minutes
val train_preds = rf_w2v14.transform(df_train1)
val predictionsAndLabels_train = train_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics = new MulticlassMetrics(predictionsAndLabels_train)   // Instantiate metrics object
val macro_F1 = F1_macro_average(metrics)

val df_val_w2v = w2vModel.transform(df_valid)               // fit word2vec trained on training data
val df_val1 = indexer.transform(df_val_w2v)
val val_preds = rf_w2v14.transform(df_val1)
val predictionsAndLabels_val = val_preds.select($"prediction", $"cuisine_idx").as[(Double, Double)].rdd   // Compute raw scores 
val metrics_val = new MulticlassMetrics(predictionsAndLabels_val)   // Instantiate metrics object
val macro_F1_val = F1_macro_average(metrics_val)

// COMMAND ----------

// <Word2Vec Results>
// maxDepth = 10 : (train_f1, valid_f1) = (0.779, 0.600)
// maxDepth = 12 : (train_f1, valid_f1) = (0.899, 0.628)
// maxDepth = 14 : (train_f1, valid_f1) = (0.963, 0.629)
// maxDepth 12 is the best setting for Word2Vec.
// With maxDepth 14, the model starts overfitting --> Miminal improvement in validation score, while +6% imporvement in training score + taking 9 more minutes to train
