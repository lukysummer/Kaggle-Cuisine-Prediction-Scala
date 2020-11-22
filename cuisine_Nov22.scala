// Databricks notebook source
val df = spark.table("train_csv")
df.printSchema()

// COMMAND ----------

display(df)

// COMMAND ----------

// convert ingredient list from String to Seq
import org.apache.spark.sql.functions._
val makeSeq : String => Array[String] = _.drop(2).dropRight(2).toLowerCase().split("', '")//.replaceAll("[^ ',a-zA-Z]", "").replaceAll("-", " ")
val makeSeqUDF = udf(makeSeq)
val df1 = df.withColumn("ingredients_list",
                        makeSeqUDF('ingredients))
            .withColumn("index", monotonicallyIncreasingId)
display(df1.select("ingredients_list"))

// COMMAND ----------

val cuisines = df1.select("cuisine")
val cuisines1 = cuisines.groupBy("cuisine").count().orderBy(desc("count"))
display(cuisines1)

// COMMAND ----------

import org.apache.spark.sql.functions._
val all_ingrs = df1.select(explode(col("ingredients_list"))).withColumnRenamed("col", "ingredient")
val all_ingrs1 = all_ingrs.groupBy("ingredient").count().orderBy(desc("count")).withColumn("index",monotonicallyIncreasingId)
display(all_ingrs1.filter($"index"<20))  // top 20 ingredients

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window 

val n_ingrs = df1.withColumn("n_ingrs", size($"ingredients_list")).orderBy(asc("n_ingrs"))
display(n_ingrs)

// COMMAND ----------

// minimum & maximum # of ingredients
import org.apache.spark.sql.functions._
val min_max_n_ingrs = n_ingrs.agg(min("n_ingrs"), max("n_ingrs"))
min_max_n_ingrs.show()

// COMMAND ----------

// Explore the distribution of number of ingredients in the data
val n_ingrs1 = n_ingrs.groupBy("n_ingrs").count().orderBy(asc("n_ingrs"))//.withColumn("index",monotonicallyIncreasingId)
display(n_ingrs1)  // top 20 ingredients

// COMMAND ----------

n_ingrs.filter($"n_ingrs">30).count()

// COMMAND ----------

val n_ingrsByCuisine = n_ingrs.select("cuisine", "n_ingrs").groupBy("cuisine").avg().orderBy("avg(n_ingrs)")
display(n_ingrsByCuisine)

// COMMAND ----------

// Dishes with only 1 ingredients
display(n_ingrs_asc.filter($"n_ingrs"===1).select("ingredients_list", "cuisine")) 

// COMMAND ----------

// CROSS VALIDATION!!
// Using my own cross validation function because running org.apache.spark.ml.tuning.CrossValidator gives me error: Exception thrown in awaitResult
import org.apache.spark.ml.feature.{StringIndexer, HashingTF, IDF}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

def crossValidator(nfolds: Int, tf_nfeats: List[Int], lr_reg: List[Double], lr_maxIter: Int) : List[Double] = {
  val indexer_ = new StringIndexer().setInputCol("cuisine").setOutputCol("cuisine_idx")
  val IDF_ = new IDF().setInputCol("ingredients_hashed").setOutputCol("ingredients_IDF")
  
  val n_records = df1.count()
  val val_size = n_records/nfolds
  val builder = List.newBuilder[Double]
  
  for (numFeatures <- tf_nfeats) {
    for (regParam <- lr_reg) {
       println("HashingTF numFeatures:   ", numFeatures, "   LR regParam:", regParam)
       var hashingTF_ = new HashingTF().setInputCol("ingredients_list").setOutputCol("ingredients_hashed")
                                       .setNumFeatures(numFeatures)
       var lr_ = new LogisticRegression().setFeaturesCol("ingredients_IDF").setLabelCol("cuisine_idx")
                                         .setMaxIter(lr_maxIter).setRegParam(regParam)
       var avg_val_acc = 0.0
       for (i <- 0 to nfolds-1) {
         var train = df1.filter($"index"<i*val_size || $"index">(i+1)*val_size)
         var valid = df1.filter($"index">i*val_size && $"index"<(i+1)*val_size)
         val pipeline = new Pipeline().setStages(Array(indexer_, hashingTF_, IDF_, lr_))
         val model = pipeline.fit(train)  
         val preds = model.transform(valid)
         val evaluator = new MulticlassClassificationEvaluator().setLabelCol("cuisine_idx").setPredictionCol("prediction")
                                                                .setMetricName("accuracy")
         val val_acc = evaluator.evaluate(preds)
         avg_val_acc += val_acc
       }
       builder += avg_val_acc/nfolds
    }
  }
 val val_acc_list = builder.result()
 return val_acc_list  
}

// COMMAND ----------

val val_acc_list = crossValidator(2, List(7000, 10000, 20000), List(0.1, 0.05, 0.01, 0.005), 50)

// COMMAND ----------

// Best Params : 
// HashingTF : numFeatures = 20000, 
// lr : regParam = 0.01

// COMMAND ----------

val Array(df_train, df_valid) = df1.randomSplit(Array[Double](0.8, 0.2), 18)

// COMMAND ----------

// Using TF-IDF Embeddings
val indexer = new StringIndexer().setInputCol("cuisine").setOutputCol("cuisine_idx")
val hashingTF = new HashingTF().setInputCol("ingredients_list").setOutputCol("ingredients_hashed")
                                       .setNumFeatures(20000)
var lr = new LogisticRegression().setFeaturesCol("ingredients_IDF").setLabelCol("cuisine_idx")
                                         .setMaxIter(50).setRegParam(0.01)
val IDF = new IDF().setInputCol("ingredients_hashed").setOutputCol("ingredients_IDF")
val pipeline = new Pipeline().setStages(Array(indexer, hashingTF, IDF, lr))
val model = pipeline.fit(df_train)  

val train_preds = model.transform(df_train)
val val_preds = model.transform(df_valid)
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("cuisine_idx").setPredictionCol("prediction")
                                                      .setMetricName("accuracy")
val train_acc = evaluator.evaluate(train_preds)
val val_acc = evaluator.evaluate(val_preds)

// COMMAND ----------

// perform Word2Vec transformation on the ingredients list
import org.apache.spark.ml.feature.{Word2Vec, Word2VecModel}

val w2vModel: Word2VecModel = new Word2Vec().
                                  setInputCol("ingredients_list").
                                  setOutputCol("ingredients_w2v").
                                  setVectorSize(350). //make this number much smaller than the size of the vocabluary (6714) 
                                  setMinCount(2).
                                  setWindowSize(10).
                                  setMaxIter(20). 
                                  fit(df_train)  // only fit

val df_after_w2v = w2vModel.transform(df_train)

// COMMAND ----------

// perform Word2Vec transformation on the ingredients list
import org.apache.spark.ml.feature.{Word2Vec, Word2VecModel}

val w2vModel: Word2VecModel = new Word2Vec().
                                  setInputCol("ingredients_list").
                                  setOutputCol("ingredients_w2v").
                                  setVectorSize(350). //make this number much smaller than the size of the vocabluary (6714) 
                                  setMinCount(2).
                                  setWindowSize(10).
                                  setMaxIter(20). 
                                  fit(df_train)  

val df_train_w2v = w2vModel.transform(df_train)

// COMMAND ----------

// perform Word2Vec transformation on the ingredients list
import org.apache.spark.ml.feature.{Word2Vec, Word2VecModel}

val w2vModel: Word2VecModel = new Word2Vec().
                                  setInputCol("ingredients_list").
                                  setOutputCol("ingredients_w2v").
                                  setVectorSize(350). //make this number much smaller than the size of the vocabluary (6714) 
                                  setMinCount(2).
                                  setWindowSize(10).
                                  setMaxIter(20). 
                                  fit(df_train)  

val df_train_w2v = w2vModel.transform(df_train)

// COMMAND ----------

val df_valid_w2v = w2vModel.transform(df_valid)

// COMMAND ----------

w2vModel.findSynonyms("meat", 10).show(false)

// COMMAND ----------

//StringIndexer for TARGET (cuisines)
import org.apache.spark.ml.feature.{StringIndexer}

val indexer = new StringIndexer().setInputCol("cuisine")
                                 .setOutputCol("cuisine_idx")
                                 .setStringOrderType("frequencyAsc")//("frequencyDesc") // order of idx doesn't matter 

val df_train_encoded = indexer.fit(df_train_w2v).transform(df_train_after_w2v)
val df_valid_encoded = indexer.fit(df_valid_w2v).transform(df_valid_after_w2v)
df_train_final.select("cuisine", "cuisine_idx").show(5, false)

// COMMAND ----------

// Using Word-to-Vec Embeddings
// Using TF-IDF Cross Validation's best lr result (regParam=0.01)
import org.apache.spark.ml.classification.LogisticRegression

val lr = new LogisticRegression()
              .setMaxIter(50)
              .setRegParam(0.01)
              .setFeaturesCol("ingredients_w2v")
              .setLabelCol("cuisine_idx")
val lrModel = lr.fit(df_encoded)
val trainingSummary = lrModel.summary
val train_acc = trainingSummary.accuracy
println("training accuracy: ", train_acc*100)

// COMMAND ----------

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val preds = lrModel.transform(df_valid_encoded)
val evaluator = new MulticlassClassificationEvaluator()
                      .setLabelCol("cuisine_idx")
                      .setPredictionCol("prediction")
                      .setMetricName("accuracy")
val val_acc = evaluator.evaluate(preds)

// COMMAND ----------

// Nov 22 2020
// TF-IDF & Word2Vec Comparison :
// TF-IDF is the winner.
// TF-IDF has higher training accuracy (91.4%) and higher validation accuracy (75.8%)
// Word2Vec has much lower training accuracy (77.9%) and lower validation accuracy (50.0%)
// Next Step : apply string pre-processing (check for punctuations, numbers, etc to remove)
