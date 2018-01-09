---
layout: post
title: Create a Spark DataFrame from Pandas or NumPy with Arrow
---

If you are a Pandas or NumPy user and have ever tried to create a Spark DataFrame from local
data, you might have noticed that it is an unbearably slow process. In fact, the time it takes
to do so usually prohibits this from any data set that is at all interesting. Starting from
Spark 2.3, the addition of [SPARK-22216][1] enables creating a DataFrame from Pandas using
Arrow to make this process much more efficient. You can now transfer large data sets to Spark
from your local Pandas session almost instantly and also be sure that your data types are
preserved. This post will demonstrate a simple example of how to do this and walk through the
Spark internals of how it is accomplished.

## A simple example to create a DataFrame from Pandas

For this example, we will generate a 2D array of random doubles from NumPy that is 1,000,000 x 10.
We will then wrap this NumPy data with Pandas, applying a label for each column name, and use this
as our input into Spark.

```python
import pandas as pd
import numpy as np

data = np.random.rand(1000000, 10)

pdf = pd.DataFrame(data, columns=list("abcdefghij"))
```

To input this data into Spark with Arrow, we first need to enable it with the below config. This
could also be included in `spark-defaults.conf` to be enabled for all sessions. Spark simply
takes the Pandas DataFrame as input and converts it into a Spark DataFrame which is distributed
across the cluster. Using Arrow, the schema is automatically transferred to Spark and data type
information will be retained, but you can also manually specify the schema to override if desired.

Assuming an existing Spark session `spark`
 
```python
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

df = spark.createDataFrame(pdf)
```

That's all there is to it! The Pandas DataFrame will be sliced up according to the number from
`SparkContext.defaultParallelism()` which can be set by the conf "spark.default.parallelism" for
the default scheduler. Depending on the size of the data you are importing to Spark, you might
need to tweak this setting.

The above can be found as a notebook gist [here][2] to try out for yourself.

## How it Works Behind the Scenes

The code path for this is pretty straight-forward and boils down to just a few key steps. All the
work is done in `SparkSession._create_from_pandas_with_arrow` from session.py, which is invoked
from `createDataFrame` after the input is found to be a Pandas DataFrame and Arrow is enabled.

1. Slice the Pandas DataFrame into chunks according to the number for default parallelism

2. Convert each chunk of Pandas data into an Arrow `RecordBatch`

3. Convert the schema from Arrow to Spark

4. Send the `RecordBatch`es to the JVM which become a `JavaRDD[Array[Byte]]`

5. Wrap the JavaRDD with the Spark schema to create a `DataFrame`

Let's look at these steps in a bit more detail to examine performance. First, slicing the Pandas
DataFrame is a cheap operation because it only uses references to the original data and does not
make copies. Converting the slices to Arrow record batches will end up copying the data since it
came from slices, but it is efficiently copied as chunks. Arrow can perform zero-copy conversions
to/from Pandas data and will do so automatically when it is able to safely reference the data.

Step 3 will create a Spark schema from Arrow schema, which is a simple mapping. Arrow has detailed
type definitions and supports all types available in Spark, however Spark only supports ya subset
of Arrow types, so you might need to be careful what you are importing. For example a union type
is supported in Arrow, but not Spark. At the time of writing this `MapType` and `StructType` are
fully supported, see the Spark documentation for more info.

Step 4 is where the Arrow data is sent to the JVM. This is necessary in actualizing the DataFrame
and will allow Spark to perform SQL operations completely within the JVM. Here the Arrow record
batches are written to a temporary file in `SparkContext._serialize_to_jvm` where they are read
back in chunks by the JVM and then parallelized to an RDD. Writing to a temporary file was done
to meld with existing code and is definitely much better than transferring the data
over a call with Py4J. In practice, this works pretty well and doesn't seem to be much of a
bottleneck and I'm not sure if setting up a local socket to send the data would do better, but
could be an area to check out in the future.

With all the above complete, the final step is done in `ArrowConverters.toDataFrame` which maps
the partitions of the `JavaRDD[Array[Byte]]` containing the Arrow record batches to an
`InternalRow` iterator and uses that along with the schema to construct the DataFrame.

## Performance Comparison with Arrow Disabled

Here is a few benchmarks of comparing the wall-clock time of calling `createDataFrame` with and
without Arrow enabled. The data used is random doubles similar to the example above, the column
_Size_ below is the total number of double values transferred. The runs were done on laptop in
Spark local mode with default Spark settings, each timing is the best of 3 consecutive iterations.

Size       | With Arrow | Without Arrow   
---------- | ---------- | ------------- 
50,000     | 14.2 ms    | 334 ms          
100,000    | 15.6 ms    | 643 ms          
500,000    | 21.9 ms    | 3.13 s          
1,000,000  | 29.6 ms    | 6.35 s        
5,000,000  | 107 ms     | 31.5 s         
10,000,000 | 245 ms     | 63 s           

I won't get into the details of the code path of when Arrow is disabled, but there are a few
reasons that make it inefficient. First, Spark does not look at the Pandas DataFrame to get
data type information, it tries to infer itself. It can not make use of NumPy data chunks and
must iterate over each record and read each value as a Python object. When it prepares the data
to send to the JVM, it must serialize each scalar value in the pickle format. Finally, once on
the JVM, it goes through another set of conversions to apply the proper Scala type.

_Download this [notebook][3] to try out the above examples or [here][2] for the gist_
 
[1]: https://issues.apache.org/jira/browse/SPARK-20791 
[2]: https://gist.github.com/BryanCutler/bc73d573b7e46a984ff8b6edf228e298 
[3]: https://gist.github.com/BryanCutler/bc73d573b7e46a984ff8b6edf228e298#file-pyspark_createdataframe_with_arrow-ipynb
