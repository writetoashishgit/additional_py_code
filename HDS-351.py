# !/usr/bin/python
"""
Comp set POC

"""

# Standard Python package imports
from os.path import join
from socket import gethostname
import numpy as np
import pandas as pd

# PySpark
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler
# from pyspark.ml.feature import BucketedRandomProjectionLSH

# Scikit-learn
from sklearn.neighbors import NearestNeighbors


def main():

    # Initialise Spark session and analysis parameters
    spark, conf_dict = initialise()

    # Run some simple tests...
    if conf_dict['local_mode']:
        test_add_haversine_distances()

    # Import data. Any filters on the underlying data should be applied here.
    df_stage_1 = import_data(spark, conf_dict)

    # Calculate average yield and average-ADR for each hotel - let's ignore booking volumes and time for the time being
    df_stage_2 = apply_basic_aggregations(df_stage_1)

    # Analyse competitive set at the city grain and write outputs to city_code-partitioned parquet
    analyse_comp_sets_per_city(spark, conf_dict, df_stage_2)

    # Additional singular output to csv, taking advantage of partition discovery on city_code
    df_comp_sets = spark.read.parquet(conf_dict['output_path_parquet'])
    df_comp_sets \
        .drop('features', 'scaledFeatures') \
        .repartition(1) \
        .write \
        .mode('overwrite') \
        .csv(conf_dict['output_path_csv'], header=True)

    # Additional output to Hive table
    if not conf_dict['local_mode']:
        df_comp_sets \
            .write \
            .saveAsTable(name=conf_dict['output_hive_table'],
                         mode='overwrite',
                         partitionBy='city_code',
                         )

    # Finish
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

    # Initialise
    conf_dict = dict()

    # Flag local mode for testing
    conf_dict['local_mode'] = gethostname() != cluster_hostname

    # Spark application name
    conf_dict['app_name'] = 'HDS-351'

    # Start the Spark session
    if conf_dict['local_mode']:
        spark = SparkSession \
            .builder \
            .appName(conf_dict['app_name']) \
            .config('spark.sql.parquet.binaryAsString', True) \
            .config('spark.driver.host', 'localhost') \
            .getOrCreate()
    else:
        spark = SparkSession \
            .builder \
            .appName(conf_dict['app_name']) \
            .config('spark.sql.parquet.binaryAsString', True) \
            .config('spark.sql.shuffle.partitions', 2000) \
            .config('spark.dynamicAllocation.enabled', True) \
            .config('spark.dynamicAllocation.maxExecutors', 4) \
            .config('spark.executor.cores', 1) \
            .getOrCreate()

    # Define source data paths
    conf_dict['output_base_path'] = 'HDS-351_comp_set_analysis_outputs'
    if not conf_dict['local_mode']:
        conf_dict['output_base_path'] = join('hdfs:///user/hoiyutang/', conf_dict['output_base_path'])

    # Set analysis output paths
    conf_dict['output_path_parquet'] = \
        join(conf_dict['output_base_path'], 'parquet')
    conf_dict['output_path_csv'] = join(conf_dict['output_base_path'], 'csv_dump')
    conf_dict['output_hive_table'] = 'sbx_dst.hyt_comp_set_analysis'

    # # Set Spark checkpoint location
    # spark.sparkContext.setCheckpointDir(join(conf_dict['output_base_path'], 'checkpoints'))

    # Set size of each competitor set
    conf_dict['comp_set_size'] = 10

    return spark, conf_dict


def import_data(spark, conf_dict):
    """
    Read bookings/transactions data and apply any filters (city, time period etc.) as appropriate
    :param spark:
    :param conf_dict:
    :return:
    """

    if conf_dict['local_mode']:
        df_stage_1 = \
            spark.read.parquet('HDS-351-stage-1-sample-data') \
                .filter('created_date >= "2018-05-01"')
    else:
        with open('HDS-351-stage-1.sql') as fr:
            stage_1_sql = fr.read()

        df_stage_1 = spark.sql(stage_1_sql)

    return df_stage_1


def apply_basic_aggregations(df):
    """
    Take all transacations and apply some basic aggregations at the harp_property_id_no grain

    :param df:
    :return:
    """
    # Where for a given harp ID, hotel chain, lat/long, hotel_name has changed over the course of the analysis window,
    # simply take the most recent hotel_name
    df_current_hotel_names = \
        df \
            .groupBy('harp_property_id_no',
                     'hotel_name',
                     'hotel_chain_name',
                     'city_code',
                     'x',
                     'y',
                     ) \
            .agg(F.max('created_date').alias('max_created_date')) \
            .withColumn('hotel_name_rank',
                        F.dense_rank().over(Window
                                            .partitionBy(F.col('harp_property_id_no'))
                                            .orderBy(F.col('max_created_date').desc()))) \
            .filter('hotel_name_rank = 1') \
            .drop('hotel_name_rank', 'max_created_date')

    # Replace hotel names with most current iteration and aggregate down to the individual property grain
    return \
        df \
            .drop('hotel_name') \
            .join(df_current_hotel_names,
                  on=['harp_property_id_no',
                      'hotel_chain_name',
                      'city_code',
                      'x',
                      'y', ]) \
            .filter('star_rating is not null AND length(star_rating) > 1') \
            .withColumn('star_rating_number',
                        F.substring('star_rating', 0, 1).cast('integer')) \
            .groupBy('harp_property_id_no',
                     'hotel_name',
                     'hotel_chain_name',
                     'city_code',
                     'x',
                     'y',
                     ) \
            .agg(F.count(F.lit(1)).alias('num_bookings'),
                 F.max('star_rating').alias('star_rating'),
                 F.max('star_rating_number').alias('star_rating_number'),
                 F.avg('yield').alias('avg_yield'),
                 F.avg('adr').alias('avg_adr'),
                 )


def analyse_comp_sets_per_city(spark, conf_dict, df):
    """
    For each city_code in the Spark dataframe df, create a comp set for each hotel within each city

    :param spark:
    :param conf_dict:
    :param df:
    :return:
    """

    # Determine unique cities in the dataset
    cities = {i.city_code for i in df.select('city_code').distinct().collect()}

    # Iterate over each unique city and calculate the comp sets per hotel. Although we could easily avoid iterations
    # and calculate the comp sets via knn en mass, this avoids any possibility of hotels from different cities being
    # included in the same comp set. Additionally, having such a large variance in lat/lon coordinates will highly
    # compress those feature dimensions and end up giving greater importance to star rating and adr
    for i, city in enumerate(cities):

        # Apply some data cleansing on lat/long coordinates
        df_cleansed = clean_geo_data(df.filter('city_code = "{}"'.format(city)))

        # TODO: Better yet, create a CompSet class with methods encompassing the end-to-end comp set creation per city
        # Build pipeline to transform/scale features at the city grain. Drop the city_code field since we'll manually
        # add this as a partitioning field after comp set creation/analysis
        df_stage_3 = transform_data_in_pipeline(df_cleansed.drop('city_code'))

        # Calculate k-nearest neighbours for all hotels to create a competitive set
        # KNN using scikit-learn for an exact NN calculation
        # NOTE: this relies upon caching the data locally and will not take advantage of Spark distributed computing
        df_stage_4 = create_comp_set(spark, conf_dict, df_stage_3)

        # # KNN using Spark Locality-Sensitive Hashing to find approx nearest neighbours
        # # NOTE: pyspark.ml.feature.BucketedRandomProjectionLSH is not implemented in PySpark 2.1.0
        # # TODO: Test accuracy of LSH-approx NN
        # knn_lsh(df_stage_3.filter('city_code = "{}"'.format(city))

        # Since the final dataframe of comp sets per city is expected to be small, let's shuffle everything into a
        # single partition and save to parquet (manually partitioned by city_code for future automatic partition
        # discovery)
        df_stage_4 \
            .repartition(1) \
            .write \
            .mode('overwrite') \
            .parquet(join(conf_dict['output_path_parquet'], 'city_code={}'.format(city)))

    return


def create_comp_set(spark, conf_dict, df):
    """
    Use k-nearest neighbour algorithm from the Scikit-learn package and return indices and distances to nearest
    neighbours for all vectors in df

    :param conf_dict:
    :param df:
    :return:
    """

    var_comp_set_size = conf_dict['comp_set_size']

    # Collect feature vectors into memory and convert vectors to numpy array
    df_vectors = df \
        .select('harp_property_id_no',
                'hotel_name',
                'scaledFeatures',
                ) \
        .cache().collect()
    np_scaled_features = np.array([i['scaledFeatures'] for i in df_vectors])
    np_harp_property_id_no = np.array([i['harp_property_id_no'] for i in df_vectors])

    # Execute knn search to create the comp sets
    dist, ind = apply_skl_knn(k=var_comp_set_size,
                              feature_vector=np_scaled_features)

    # Manual inspection
    """    
    df \
    .filter(F.col('harp_property_id_no')
            .isin([vec['harp_property_id_no'] for i, vec in enumerate(vecs) if i in ind[10]])) \
    .show()
    
    ind[10]
    dist[10]
    """

    # Prepare output dataframe containing all comp sets, properties, and calculate performance metrics wrt target
    # property within each comp set
    compset_df = \
        spark.createDataFrame(
            pd.DataFrame(
                {
                    'target_property_harp_ID':
                        np.array([i['harp_property_id_no'] for i in df_vectors]).repeat(var_comp_set_size),
                    'target_property_name':
                        np.array([i['hotel_name'] for i in df_vectors]).repeat(var_comp_set_size),
                    'harp_property_id_no': np_harp_property_id_no[ind].ravel(),
                    'knn_distance_metric': dist.ravel(),
                }
            )
        ) \
        .join(df, on='harp_property_id_no') \
        .join(df.select(F.col('harp_property_id_no').alias('target_property_harp_ID'),
                        F.col('avg_yield').alias('target_property_yield'),
                        F.col('num_bookings').alias('target_property_num_bookings'),
                        F.col('x').alias('target_property_longitude'),
                        F.col('y').alias('target_property_latitude'),
                        ),
              on='target_property_harp_ID') \
        \
        .withColumn('compset_yield_vs_target_property_diff', F.col('avg_yield') - F.col('target_property_yield')) \
        .withColumn('compset_yield_vs_target_property_ratio', F.col('avg_yield') / F.col('target_property_yield')) \
        .withColumn('compset_yield_performance_rank',
                    F.dense_rank().over(Window
                                        .partitionBy(F.col('target_property_harp_ID'))
                                        .orderBy(F.col('compset_yield_vs_target_property_ratio').desc()))) \
        \
        .withColumn('compset_share_of_bookings',
                    F.col('num_bookings') / F.sum('num_bookings').over(
                        Window.partitionBy(F.col('target_property_harp_ID')))) \
        .withColumn('compset_bookings_wrt_target_property_ratio',
                    F.col('num_bookings') / F.col('target_property_num_bookings')) \
        .withColumn('compset_share_of_bookings_rank',
                    F.dense_rank().over(Window
                                        .partitionBy(F.col('target_property_harp_ID'))
                                        .orderBy(F.col('compset_bookings_wrt_target_property_ratio').desc())))

    # Add distance between from target hotel to other hotels within each comp set
    compset_df = add_haversine_distances(compset_df,
                                         lat_a='y',
                                         lon_a='x',
                                         lat_b='target_property_latitude',
                                         lon_b='target_property_longitude',
                                         )

    # Re-order fields for readability and return
    return compset_df.select(
        'target_property_harp_ID',
        'target_property_name',
        'target_property_longitude',
        'target_property_latitude',
        'target_property_yield',
        'target_property_num_bookings',
        'harp_property_id_no',
        'hotel_name',
        'hotel_chain_name',
        F.col('x').alias('longitude'),
        F.col('y').alias('latitude'),
        'star_rating',
        'star_rating_number',
        'num_bookings',
        'avg_yield',
        'avg_adr',
        'features',
        'scaledFeatures',
        'knn_distance_metric',
        'distance_km',
        'compset_yield_vs_target_property_diff',
        'compset_yield_vs_target_property_ratio',
        'compset_yield_performance_rank',
        'compset_share_of_bookings',
        'compset_bookings_wrt_target_property_ratio',
        'compset_share_of_bookings_rank',
    )


def knn_lsh(df):

    brp = BucketedRandomProjectionLSH(inputCol='features',
                                      outputCol='hashes',
                                      bucketLength=2,
                                      numHashTables=3)
    model = brp.fit(df)
    model.transform(df).show()
    model \
        .approxNearestNeighbors(dataset=df,
                                key=model.transform(df).select('features').take(2)[0]['features'],
                                numNearestNeighbors=10,
                                distCol='distCol') \
        .show()

    features = np.array(df).select('features').collect()
    knn = NearestNeighbors(n_neighbors=10).fit(features)


def transform_data_in_pipeline(df):
    """

    :param df:
    :return:
    """

    # Initialise pipeline variables
    stages = []
    assembler_inputs = []

    # Assemble features vector from Spark dataframe fields
    assembler = VectorAssembler(
        inputCols=['x', 'y', 'star_rating_number', 'avg_adr'],
        outputCol='features'
    )
    stages += [assembler]
    assembler_inputs += [assembler.getOutputCol()]

    # Apply standard scaling with unit std and centroid about the mean
    scaler = StandardScaler(
        inputCol=assembler.getOutputCol(),
        outputCol='scaledFeatures'
    )
    stages += [scaler]
    assembler_inputs += [scaler.getOutputCol()]

    # Execute the pipeline
    pipeline_model = Pipeline() \
        .setStages(stages) \
        .fit(df)

    # Return the dataframe with the additional transformed features vector
    return pipeline_model.transform(df)


def apply_skl_knn(k, feature_vector):

    # Instantiate knn object
    knn = NearestNeighbors(n_neighbors=k,
                           metric='euclidean',
                           algorithm='auto',
                           )
                           # algorithm='brute',

    # Fit the model
    knn.fit(feature_vector)

    # Calculate the k-nearest neighbours for all vectors
    dist, ind = knn.kneighbors(feature_vector,
                               k,
                               return_distance=True,
                               )

    return dist, ind


def add_haversine_distances(df, lat_a, lon_a, lat_b, lon_b):
    """
    Although the Haversine distance is readily available from sklearn.metrics.pairwise.haversine_distances, the
    implementation below makes use only of in-built PySpark functions, so should be more efficient than passing the
    sklearn function in as a UDF.

    :param df:
    :param lat_a:
    :param lon_a:
    :param lat_b:
    :param lon_b:
    :return:
    """

    earth_radius_km = 6371.0

    return \
        df \
            .withColumn('dist_lat', F.radians(lat_a) - F.radians(lat_b)) \
            .withColumn('dist_lon', F.radians(lon_a) - F.radians(lon_b)) \
            .withColumn('area',
                        (F.sin(F.col('dist_lat') / 2) ** 2)
                        + (F.cos(F.radians(lat_a))
                           * F.cos(F.radians(lat_b))
                           * (F.sin(F.col('dist_lon') / 2) ** 2)
                           )
                        ) \
            .withColumn('central_angle', 2 * F.asin(F.sqrt(F.col('area')))) \
            .withColumn('distance_km', F.col('central_angle') * F.lit(earth_radius_km)) \
            .drop('dist_lat', 'dist_lon', 'area', 'central_angle')


def clean_geo_data(df):
    """
    Apply some basic data quality checks on geographical data

    """

    # Calculate the value range corresponding to the percentile limits, also ignoring any values that lie outside of
    # the allowable lat/long ranges
    percentile_lower_limit = 0.03
    percentile_upper_limit = 0.97

    percentiles = {}
    for col in ['x', 'y']:
        percentiles[col] = \
            df \
                .filter('x between -180 and 180 and y between -90 and 90') \
                .select(col) \
                .distinct() \
                .approxQuantile(col, [percentile_lower_limit, percentile_upper_limit], 0)
    # print(percentiles)

    # Apply filter to remove long tails
    return \
        df \
            .filter('x between {} and {}'.format(*percentiles['x'])) \
            .filter('y between {} and {}'.format(*percentiles['y']))


def test_add_haversine_distances():

    from pandas.testing import assert_frame_equal

    spark = SparkSession.builder \
        .master('local[2]') \
        .appName('local-testing-pyspark-session') \
        .getOrCreate()

    test_input = \
        pd.DataFrame({'id': [0, 1, 2, 3, 4, 5],
                      'x': [0.0, 0.0, 0.0, -180.0, 180.0, 0.0],
                      'y': [0.0, 90.0, -90.0, 0.0, 0.0, -90.0],
                      'target_property_latitude': [0.0, 0.0, 0.0, -180.0, 180.0, 0.0],
                      'target_property_longitude': [0.0, 90.0, -90.0, 0.0, 0.0, 90.0],
                      })

    results = \
        add_haversine_distances(spark.createDataFrame(test_input),
                                lat_a='x',
                                lon_a='y',
                                lat_b='target_property_latitude',
                                lon_b='target_property_longitude') \
            .toPandas() \
            .round(1) \
            .sort_values(by='id')

    expected_results = \
        pd.DataFrame({'id': [0, 1, 2, 3, 4, 5],
                      'x': [0.0, 0.0, 0.0, -180.0, 180.0, 0.0],
                      'y': [0.0, 90.0, -90.0, 0.0, 0.0, -90.0],
                      'target_property_latitude': [0.0, 0.0, 0.0, -180.0, 180.0, 0.0],
                      'target_property_longitude': [0.0, 90.0, -90.0, 0.0, 0.0, 90.0],
                      'distance_km': [0.0, 0.0, 0.0, 0.0, 0.0, 20015.1],
                      }) \
            .sort_values(by='id')

    assert_frame_equal(expected_results, results)

    spark.stop()


if __name__ == "__main__":
    main()

