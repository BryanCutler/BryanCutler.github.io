---
layout: post
title: Spark Cross-Validation with Mulitple Pipelines
---

Cross-Validation with Apache Spark `Pipeline`s is commonly used to tune the hyperparameters of 
stages in a `PipelineModel`. But what do you do if you want to evaluate more than one pipeline 
with different stages, e.g. using different types of classifiers? You would probably just run 
cross-validation on each pipeline separately and compare the results, which would generally
work fine. You might not know that `stages` are actually a parameter in the `PipelineModel` and
can be evaluated just like any other parameter, with a few caveats. So it is possible to put
multiple pipelines into the Spark `CrossValidator` to automatically select the best one and make
more efficient use of your data and caching. This post will show you how.

## CV Example with Two Pipelines

This use case came out of an existing JIRA at [SPARK-19979][1] and will work for pipeline
model selection with Spark ML `CrossValidator` and `TrainValidationSplit`. Let's start with an
example where I want to construct a classification pipeline, but I am not tied to any particular
classifier, just which ever one performns best under cross-validation according to my metric. 
First, I will define some pipeline stages and an empty pipeline. It is crucial to declare the
pipeline empty because the `stages` are a parameter and will be added as part of the parameter
grid.

```scala
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{PCA, VectorIndexer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}

val vectorIndexer = new VectorIndexer()
    .setInputCol("features")
    .setOutputCol("indexedFeatures")

val pca = new PCA()
    .setInputCol(vectorIndexer.getOutputCol)
    .setOutputCol("pcaFeatures")

val dt = new DecisionTreeClassifier()
    .setLabelCol("label")
    .setFeaturesCol(vectorIndexer.getOutputCol)

val lr = new LogisticRegression()
    .setLabelCol("label")
    .setFeaturesCol(pca.getOutputCol)

val pipeline = new Pipeline()
```

Now lets make an Array of stages that will correspond to the pipelines we want and build a 
separate parameter grid for each of those pipelines.

```scala
val pipeline1 = Array[PipelineStage](vectorIndexer, dt)
val pipeline2 = Array[PipelineStage](vectorIndexer, pca, lr)

val pipeline1_grid = new ParamGridBuilder()
  .baseOn(pipeline.stages -> pipeline1)
  .addGrid(dt.maxDepth, Array(2, 5))
  .build()

val pipeline2_grid = new ParamGridBuilder()
  .baseOn(pipeline.stages -> pipeline2)
  .addGrid(pca.k, Array(10, 20))
  .addGrid(lr.regParam, Array(0.1, 0.01))
  .build()

```

Here we make use of the `ParamGridBuilder` in Spark and use the method `baseOn()` to build a grid
where all param values will have a fixed pipeline. It's important to note that the `ParamGridBuilder`
simply builds an Array with all combinations of param values. This means that once we have built
a grid for each pipeline, we can just concatenate the Arrays to get the grid we want and then use
that to run cross-validation. 

```scala
val paramGrid = pipeline1_grid ++ pipeline2_grid

val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(5)

// Run cross-validation, assuming "training" data exists
val cvModel = cv.fit(training)

// Get the best selected pipeline model
val pipelineModel = cvModel.bestModel.asInstanceOf[PipelineModel]
```
The above code has the advantage over running cross-validation twice on each pipeline and manually
comparing the result because this will allow each pipeline to reuse the cached data folds. Inside
the `CrossValidator.fit` call, once the data is split into folds and divided into training and
evaluation sets, it is cached. So by combining both pipelines into a single `cv.fit`, the cached
data is reused and that can make a huge improvement in processing time.

## Details on Constructing an Optimum Param Grid

The reason we construct the param grids separately is because the logistic regression pipeline is
additionally using PCA to reduce dimensionality, while the decision tree pipeline is not. If we
were to construct just a single grid, everything would work, but it would mean that we are 
performing unnecessary work because the PCA parameters have no affect on  the decision tree which 
does not even use the resulting column. To illustrate this let's look at the output of our param grid.

```
paramGrid: Array[org.apache.spark.ml.param.ParamMap] =
Array({
    dtc_da28416bb307-maxDepth: 2,
    pipeline_2e9bf0dddb22-stages: [Lorg.apache.spark.ml.PipelineStage;@218e188e
}, {
    dtc_da28416bb307-maxDepth: 5,
    pipeline_2e9bf0dddb22-stages: [Lorg.apache.spark.ml.PipelineStage;@218e188e
}, {
    pca_d39f26a3e7dd-k: 10,
    logreg_d3b3ce8022a9-regParam: 0.1,
    pipeline_2e9bf0dddb22-stages: [Lorg.apache.spark.ml.PipelineStage;@d3e77a3
}, {
    pca_d39f26a3e7dd-k: 10,
    logreg_d3b3ce8022a9-regParam: 0.01,
    pipeline_2e9bf0dddb22-stages: [Lorg.apache.spark.ml.PipelineStage;@d3e77a3
}, {
    pca_d39f26a3e7dd-k: 20,
    logreg_d3b3ce8022a9-regParam: 0.1,
    pipeline_2e9bf0dddb22-stages: [Lorg.apache.spark.ml.PipelineStage;@d3e77a3
}, {
    pca_d39f26a3e7dd-k: 20,
    logreg_d3b3ce8022a9-regParam: 0.01,
    pipeline_2e9bf0dddb22-stages: [Lorg.apache.spark.ml.PipelineStage;@d3e77a3
})
```

Each element in the array represents a pipeline model that will be evaluated at each cross-validation
fold. There are 2 decision tree pipelines for each value of `maxDepth` we are evaluating and 4 
logistic regression pipelines for all combinations of PCA `k` and LogisticRegression `regParam`.
If instead we made one single grid including both pipelines, you can see the grid is much larger
(16 vs. 6 models) because each combination of PCA is run on the decision tree, along with the classifier
params `regParam` and `maxDepth` even if they are not for the classifer in that `PipelineModel`
being evaluated so we end up doing much more work than is really needed.

```
val dumbGrid = new ParamGridBuilder()
    .addGrid(pipeline.stages, Array(pipeline1, pipeline2))
    .addGrid(dt.maxDepth, Array(2, 5))
    .addGrid(pca.k, Array(10, 20))
    .addGrid(lr.regParam, Array(0.1, 0.01))
    .build()

// Exiting paste mode, now interpreting.

dumbGrid: Array[org.apache.spark.ml.param.ParamMap] =
Array({
    pca_d39f26a3e7dd-k: 10,
    dtc_da28416bb307-maxDepth: 2,
    logreg_d3b3ce8022a9-regParam: 0.1,
    pipeline_2e9bf0dddb22-stages: [Lorg.apache.spark.ml.PipelineStage;@218e188e
}, {
    pca_d39f26a3e7dd-k: 10,
    dtc_da28416bb307-maxDepth: 2,
    logreg_d3b3ce8022a9-regParam: 0.1,
    pipeline_2e9bf0dddb22-stages: [Lorg.apache.spark.ml.PipelineStage;@d3e77a3
}, {
    pca_d39f26a3e7dd-k: 10,
    dtc_da28416bb307-maxDepth: 5,
    logreg_d3b3ce8022a9-regParam: 0.1,
    pipeline_2e9bf0dddb22-stages: [Lorg.apache.spark.ml.PipelineStage;@218e188e
}, {
    pca_d39f26a3e7dd-k: 10,
    dtc_da28416bb307-maxDepth: 5,
    logreg_d3b3ce8022a9-regParam: 0.1,
    pipeline_2e9bf0dddb22-stages: [Lorg.apache.spark.ml.PipelineStage;@d3e77a3
}, {
    pca_d39f26a3e7dd-k: 10,
    dtc_da28416bb307-maxDepth: 2,
    logreg_d3b3ce8022a9-regParam: 0.01,
    pipeline_2e9bf0dddb22-stages: [Lorg.apache.spark.ml.PipelineStage;@218e188e
}, {
    pca_d39f26a3e7dd-k: 10,
    dtc_da28416bb307-maxDepth: 2,
    logreg_d3b3ce8022a9-regParam: 0.01,
    pipeline_2e9bf0dddb22-stages: [Lorg.apache.spark.ml.PipelineStage;@d3e77a3
}, {
    pca_d39f26a3e7dd-k: 10,
    dtc_da28416bb307-maxDepth: 5,
    logreg_d3b3ce8022a9-regParam: 0.01,
    pipeline_2e9bf0dddb22-stages: [Lorg.apache.spark.ml.PipelineStage;@218e188e
}, {
    pca_d39f26a3e7dd-k: 10,
    dtc_da28416bb307-maxDepth: 5,
    logreg_d3b3ce8022a9-regParam: 0.01,
    pipeline_2e9bf0dddb22-stages: [Lorg.apache.spark.ml.PipelineStage;@d3e77a3
}, {
    pca_d39f26a3e7dd-k: 20,
    dtc_da28416bb307-maxDepth: 2,
    logreg_d3b3ce8022a9-regParam: 0.1,
    pipeline_2e9bf0dddb22-stages: [Lorg.apache.spark.ml.PipelineStage;@218e188e
}, {
    pca_d39f26a3e7dd-k: 20,
    dtc_da28416bb307-maxDepth: 2,
    logreg_d3b3ce8022a9-regParam: 0.1,
    pipeline_2e9bf0dddb22-stages: [Lorg.apache.spark.ml.PipelineStage;@d3e77a3
}, {
    pca_d39f26a3e7dd-k: 20,
    dtc_da28416bb307-maxDepth: 5,
    logreg_d3b3ce8022a9-regParam: 0.1,
    pipeline_2e9bf0dddb22-stages: [Lorg.apache.spark.ml.PipelineStage;@218e188e
}, {
    pca_d39f26a3e7dd-k: 20,
    dtc_da28416bb307-maxDepth: 5,
    logreg_d3b3ce8022a9-regParam: 0.1,
    pipeline_2e9bf0dddb22-stages: [Lorg.apache.spark.ml.PipelineStage;@d3e77a3
}, {
    pca_d39f26a3e7dd-k: 20,
    dtc_da28416bb307-maxDepth: 2,
    logreg_d3b3ce8022a9-regParam: 0.01,
    pipeline_2e9bf0dddb22-stages: [Lorg.apache.spark.ml.PipelineStage;@218e188e
}, {
    pca_d39f26a3e7dd-k: 20,
    dtc_da28416bb307-maxDepth: 2,
    logreg_d3b3ce8022a9-regParam: 0.01,
    pipeline_2e9bf0dddb22-stages: [Lorg.apache.spark.ml.PipelineStage;@d3e77a3
}, {
    pca_d39f26a3e7dd-k: 20,
    dtc_da28416bb307-maxDepth: 5,
    logreg_d3b3ce8022a9-regParam: 0.01,
    pipeline_2e9bf0dddb22-stages: [Lorg.apache.spark.ml.PipelineStage;@218e188e
}, {
    pca_d39f26a3e7dd-k: 20,
    dtc_da28416bb307-maxDepth: 5,
    logreg_d3b3ce8022a9-regParam: 0.01,
    pipeline_2e9bf0dddb22-stages: [Lorg.apache.spark.ml.PipelineStage;@d3e77a3
})
``` 

### Further Optimizations

One final note is that even with the above method, there is still some duplicated work due to the 
common `VectorIndexer` stage that exists in both pipelines. So for a given cross-validation fold, the 
`VectorIndexer` transforms the same data for each `PipelineModel` being evaluated. There is no easy 
workaround for this except for brining the `VectorIndexer` out of the pipelines, but that kind of
defeats the purpose of constructing pipelines in the first place. Optimizing pipelines to reduce
this duplicated work is currently a work in progress that can be tracked on [SPARK-19071][2].

[1]: https://issues.apache.org/jira/browse/SPARK-19979
[2]: https://issues.apache.org/jira/browse/SPARK-19071
