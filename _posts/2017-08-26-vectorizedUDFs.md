---
layout: post
title: Vectorized UDFs in PySpark
---

With the introduction of Apache Arrow in Spark, it makes it possible to evaluate Python UDFs as
vectorized functions. In addition to the performance benefits from vectorized functions, it also
opens up more possibilities by using Pandas for input and output of the UDF. This post will show
some details of on-going work I have been doing in this area and how to put it to use.

## Some Context

I have been experimenting with using Arrow for Python UDFs in Spark for a while now, and recently
put this work into [SPARK-21404][1] and the pull-request [#18659][2]. It is unclear if this will
end up merged in Spark, so for now you will have to apply the patch manually. Currently it is in
a usable state, but not production ready so no guarantees and use at your own risk. There is also
some great work done in this same vein by Li Jin, who is using Arrow to provide a split-apply-merge
workflow with Pandas UDFs, see [SPARK-20396][3]. Finally, keep a watch on the Spark SPIP for
vectorized UDFs [here][4].

I have prepared a sample notebook that demonstrates how to use this optimization, a performance
comparison to standard Python UDFs, and making better use of Pandas and Numpy in your UDFs. You
can download it as a gist [here][5]. 

## How to Enable

_Updated November 8, 2017_

There is no longer any configuration setting to enable vectorized UDFs, instead you need to declare
your function to Spark as `pandas_udf`:

```
# Wrap the function "func"
pandas_udf(func, DoubleType())

# Use a decorator
@pandas(returnType=DoubleType())
def func(x):
    ...
    return y
```

With that done, you will want to make sure your UDFs are in a vectorized form. Most mathematical
functions can be written this way and there are plenty of resources on how to. The API for this is
simple, all inputs to the function will be `Pandas.Series` and the output should be a single
`Pandas.Series` or Numpy array of the same length as the inputs. For example, in the vectorized
function below, inputs `a` and `b` are series. The output `c` is also a series that is the result
of a vectorized addition.

```python
def func(a, b):
    c = a + b
    return c
```

This vectorized function can be then be made into a Python UDF exactly the same way you would
normally define a `udf` in Spark and can then be expressed as a column in Spark SQL with the return
type as specified, for instance assuming a `DataFrame` "df" with existing columns "a" and "b":

```python
func_udf = pandas_udf(func, DoubleType())
df.withColumn("c", func_udf(col("a"), col("b")))
```

## Behind the Scenes

There is not much new that had to be added to Spark to get this working. Because it is an optional
conf and there could now be 2 kinds of UDFs, some indicators had to be added to coordinate between
the Java `PythonRDD` and the Python worker what type of UDF was being used. Other than that, the
process is pretty simple: 

1) Partitions are converted to one or more Arrow record batches containing the columns that are
inputs to the function. This is done as an iterator in `ArrowConverters.toPayloadIterator` so that
once the Python worker process is started, batches can be iteratively transferred over a socket.

2) A new Python serializer `pyspark.serializers.ArrowPandasSerializer` was made to receive the
batch iterator, load the next batch as Arrow data, and create a `Pandas.Series` for each
`pyarrow.Column`. 

3) The input series are then applied to the function and the resulting series is dumped to the
serializer that will convert it to Arrow data and send it back over the socket.

4) Upon receiving the resulting Arrow data, `ArrowConverters.fromPayloadIterator` will transform it
into Spark SQL `Row`s where it will become the defined UDF column in the `Dataset`.

### The Spark SQL Physical Plan

The entire process is controlled by the Spark SQL physical plan `ArrowEvalPythonExec`. When
executed, this will map the partitions of the Dataset by first creating a row iterator of the
inputs to one or more UDFs, sending that iterator to a Python process, and joining the resulting
row iterator with the original. The big drawback here is that this is still dominated by row-based
operations which creates a huge bottleneck because now we are dealing with columnar data, and it
is slow and painful to have to convert between these formats.

### Arrow Columnar Batches in Spark

Ideally for this type of workflow, we would not have to deal with any row operation. However, that
will probably not be a reality until Spark has an internal columnar engine - which has been
discussed but probably a long way off. For now, making use of the existing `ColumnarBatch`
functionality with Arrow vectors can help out some. This would allow Arrow data to be read directly
to Spark `ColumnVector` and grouped into a batch that can be used as a row iterator. This has
the benefit of allowing the data returned from the UDF to be consumed by the `ArrowEvalPythonExec`
execution without any further copying.

## Benefits of Vectorization

### Faster

It is my opinion that once you get used to writing vectorized functions, it is a more natural
way to express computations, especially for statistics and machine learning applications. The most
obvious benefit of doing this is for performance gains by avoiding loops and pushing down
calculations to lower-level optimized code. The performance increase will vary depending on your
specific use case, but from some simple tests done locally on my laptop, I have been seeing a
speedup of ~3.7x. You can see this example test done in the [notebook][5]. That speedup is not
as dramatic as the one seen in my previous [post][6] with using `toPandas()` with Arrow, but
if your UDFs are being run on large datasets day in and day out, this will definitely make a
difference. It is also still a work-in-progress and I will keep trying to tweak this optimization
to get every last drop of performance out.

### Make better use of Pandas and Numpy

Using Pandas and Numpy for data science applications is practically de-facto, and not being able
to use these awesome packages is just sad! Currently, Python UDFs in Spark only work with scalar
values of standard python types and even if you try to use a numpy function, you will have to cast
it back to Python or Spark will not be able to pickle it, see [SPARK-12157][7]. Not so for
vectorized  UDFs with Arrow. Once enabled, your function inputs will be `Pandas.Series` and you are
able to make full use of Pandas/Numpy functionality on these, then the return value can also be a
series or numpy array of the same length. Here is a simple example of a function that will only
work with vectorization enabled:

```python
def sample(label):
    """
    Sample selected data from a Poisson distribution
    :param label: Pandas.Series of data labels
    """

    # use numpy to initialze an empty array
    p = pd.Series(np.zeros(len(label)))
    
    # use pandas to select data matching label "0"
    idx0 = label == 0

    # sample from numpy and assign to the selected data
    p[idx0] = np.random.poisson(7, len(idx0))

    # return the pandas series
    return p
```

This is just a contrived example to show what you can do, there are obvious better ways to
accomplish this. The important take away is that this can allow you to start with a small
scale application using standard Python packages. Then later, scale up to a large cluster for big
data with Spark and still be able utilize the same functions without a bunch of rewriting.

### Try it out

If this type of functionality could be useful to you, I urge you to try out the [patch][8] from
[SPARK-21404][1] and vote or watch this issue. Also, please participate in the [SPIP][4] discussion
with any feedback from real use cases is always a huge help.

_Updated November 8, 2017 to reflect API changes_

### Good News!

Vectorized UDFs have been merged into Spark along with groupby-apply with Pandas DataFrames from
[SPARK-20396][3]. Some details are still in the works, but be sure to look for this functionality
in the Spark 2.3 release!


[1]: https://issues.apache.org/jira/browse/SPARK-21404
[2]: https://github.com/apache/spark/pull/18659
[3]: https://issues.apache.org/jira/browse/SPARK-20396
[4]: https://issues.apache.org/jira/browse/SPARK-21190
[5]: https://gist.github.com/BryanCutler/0b0c820c1beb5ffc40618c462912195f
[6]: https://bryancutler.github.io/toPandas/
[7]: https://issues.apache.org/jira/browse/SPARK-12157
[8]: https://patch-diff.githubusercontent.com/raw/apache/spark/pull/18659.diff
