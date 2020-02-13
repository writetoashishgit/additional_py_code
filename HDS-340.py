# !/usr/bin/python
"""
HDS-340 - Frequency of un/available CNR Rates at all Properties

### Execution benchmarks
~10 minutes for 27th May 2019 @2500 partitions, 10 executors, 2 cores

"""

from os.path import join
from socket import gethostname

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


def main():

    # Initialise Spark session and analysis parameters
    spark, conf_dict = initialise()

    # The main analysis section
    df = analyse(spark, conf_dict)

    # Save results
    df \
        .repartition('year', 'month', 'day') \
        .write.mode('overwrite').partitionBy('year', 'month', 'day').parquet(conf_dict['output_path'])

    # Summarise results
    summarise(spark, conf_dict)

    spark.stop()

    return


def initialise():
    """
    Define PySpark application parameters and start Spark session
    :return:
    spark - the application object
    conf_dict - python dictionary of parameters
    """

    # Define our cluster hostname
    cluster_hostname = 'ingest-appcloud.cazena.internal'

    conf_dict = dict()

    # Flag local mode for testing
    conf_dict['local_mode'] = gethostname() != cluster_hostname

    # Spark application name
    conf_dict['app_name'] = 'HDS-340'

    # Start the Spark session
    if conf_dict['local_mode']:
        spark = SparkSession \
            .builder \
            .appName(conf_dict['app_name']) \
            .config('spark.sql.parquet.binaryAsString', True) \
            .getOrCreate()
    else:
        spark = SparkSession \
            .builder \
            .appName(conf_dict['app_name']) \
            .config('spark.sql.parquet.binaryAsString', True) \
            .config('spark.sql.shuffle.partitions', 20000) \
            .config('spark.dynamicAllocation.enabled', True) \
            .config('spark.dynamicAllocation.maxExecutors', 4) \
            .config('spark.executor.cores', 1) \
            .getOrCreate()

    # Define source data paths
    if conf_dict['local_mode']:
        conf_dict['src_hotel_searches'] = '/Users/u020hxt/Downloads/propertyAvailabilityRS'
        conf_dict['src_rates_lookup_deduped'] = '/Users/u020hxt/Downloads/sw_rate_to_top_client_lookup_deduped'
        conf_dict['output_base_path'] = '/Users/u020hxt/Downloads/'
    else:
        # Note - read from hdfs parquet files rather than Hive tables since filtering on integer partition fields is not
        # supported in the latter
        conf_dict['src_hotel_searches'] = '/users/shared_data/dst/hotels/propertyAvailabilityRS/'
        conf_dict['src_rates_lookup_deduped'] = '/user/hoiyutang/sw_rate_to_top_client_lookup_deduped'
        conf_dict['output_base_path'] = 'hdfs:///user/hoiyutang/'

    # Set analysis output path
    conf_dict['output_path'] = join(conf_dict['output_base_path'], 'HDS-340_rate_counts_per_search')

    # Set Spark checkpoint location
    spark.sparkContext.setCheckpointDir(join(conf_dict['output_base_path'], 'checkpoints'))

    # Define the schema for the GRAMPA logs in src_hotel_searches
    conf_dict['schema'] = T.StructType([
        T.StructField('year', T.IntegerType()),
        T.StructField('month', T.IntegerType()),
        T.StructField('day', T.IntegerType()),
        T.StructField('res_sessionID', T.StringType()),
        T.StructField('hotel_id', T.StringType()),
        T.StructField('id', T.StringType()),
        T.StructField('ratePlan_tpaExtensions_labels_label',
                      T.ArrayType(
                          T.StructType([T.StructField('id', T.StringType()),
                                        T.StructField('type', T.StringType()),
                                        T.StructField('value', T.StringType()),
                                        ])),
                      )
    ])

    return spark, conf_dict


def analyse(spark, conf_dict):
    """

    :param spark: the Spark session object
    :param conf_dict: parameter dictionary
    :return:
    """
    # Read data from sources
    df_searches = spark.read.schema(conf_dict['schema']).parquet(conf_dict['src_hotel_searches'])
    df_searches.printSchema()
    df_rates_lookup = spark.read.parquet(conf_dict['src_rates_lookup_deduped'])

    # Select subset of required fields. Join onto rate lookup table and de-dupe
    df_stage_1 = df_searches \
        .filter('year = 2019 AND month = 5 AND day BETWEEN 24 AND 30') \
        .select('year',
                'month',
                'day',
                'res_sessionID',
                'hotel_id',
                df_searches['id'].alias('rate_id'),
                'ratePlan_tpaExtensions_labels_label',
                ) \
        .join(df_rates_lookup, on='rate_id', how='left') \
        .select('year',
                'month',
                'day',
                'res_sessionID',
                'rate_id',
                'hotel_id',
                F.coalesce(F.col('new_rate_bucket'), F.lit('OTHER')).alias('new_rate_bucket'),
                'ratePlan_tpaExtensions_labels_label',
                ) \
        .distinct() \
        .checkpoint()

    # Explode ratePlan_tpaExtensions_labels_label to extract rate sorted rank and rate type
    df_stage_2 = df_stage_1 \
        .withColumn('expl', F.explode('ratePlan_tpaExtensions_labels_label')) \
        .withColumn('label_length', F.size(F.col('ratePlan_tpaExtensions_labels_label'))) \
        .filter('(expl.id is null AND label_length = 1) '
                'OR (expl.id = "CWT_RATE_SORTED_RANK" AND expl.type = "CWT_RATE_SORTED_RANK") '
                'OR (expl.id = "CWT_RATE_TYPE" AND expl.type = "CWT_RATE_TYPE") '
                ) \
        .withColumn('label_id', F.col('expl.id')) \
        .withColumn('label_value', F.col('expl.value')) \
        .groupBy('year',
                 'month',
                 'day',
                 'res_sessionId',
                 'hotel_id',
                 'rate_id',
                 'new_rate_bucket',
                 'ratePlan_tpaExtensions_labels_label',
                 ) \
        .pivot('label_id', ['CWT_RATE_TYPE', 'CWT_RATE_SORTED_RANK']) \
        .agg(F.first(F.col('label_value'))) \
        .checkpoint()

    # Aggregations
    df_stage_3 = df_stage_2 \
        .withColumn('top3_rate_flag',
                    F.when(F.col('CWT_RATE_SORTED_RANK').cast('float').between(1, 3), 1).otherwise(0)) \
        .groupBy('year',
                 'month',
                 'day',
                 'res_sessionId',
                 'hotel_id',
                 'new_rate_bucket',
                 'CWT_RATE_TYPE',
                 'top3_rate_flag',
                 ) \
        .agg(F.count(F.lit(1)).alias('responses'))

    return df_stage_3


def summarise(spark, conf_dict):
    """
    Queries to summarise the base dataframe as various grains suitable for business consumption

    :param spark: Spark session object
    :param conf_dict: parameter dictionary
    :return:
    """

    groupby_fields = ['year',
                      'month',
                      'day',
                      'CWT_RATE_TYPE',
                      'top3_rate_flag',
                      ]

    # Retrieve analysis results
    df = spark.read.parquet(conf_dict['output_path'])

    # Apply final aggregations and dump to a single csv
    df \
        .groupBy(groupby_fields) \
        .agg(F.sum('responses').alias('total_responses')) \
        .orderBy(groupby_fields) \
        .repartition(1) \
        .write \
        .mode('overwrite') \
        .csv(path=conf_dict['output_path'] + '_summary', header=True)


if __name__ == "__main__":
    main()

