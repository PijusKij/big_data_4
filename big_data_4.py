from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, unix_timestamp, to_timestamp, when, sum as _sum
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
import math
from google.colab import drive
from pyspark.sql.functions import udf

# reading data
drive.mount('/content/drive')
file_path = "/content/drive/My Drive/aisdk-2025-05-04.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# cleaning data
df1 = df.select("# Timestamp", "MMSI", "Latitude", "Longitude")

df1 = df1.withColumnRenamed("# Timestamp", "timestamp_str")
df1 = df1.withColumn("timestamp", to_timestamp(col("timestamp_str"), "dd/MM/yyyy HH:mm:ss"))
df1 = df1.withColumn("timestamp_unix", unix_timestamp(col("timestamp")))

df1 = df1.filter(
    col("Latitude").isNotNull() & col("Longitude").isNotNull() &
    col("timestamp_unix").isNotNull() &
    col("Latitude").between(-90, 90) &
    col("Longitude").between(-180, 180)
)

df1 = df1.dropDuplicates(["MMSI", "timestamp_unix"])

# function for haversine distance calculations
def haversine(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return 0.0
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2.0)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

haversine_udf = udf(haversine, DoubleType())

#lagging time and coordinates
w = Window.partitionBy("MMSI").orderBy("timestamp_unix")

df1 = df1.withColumn("prev_lat", lag("Latitude").over(w))
df1 = df1.withColumn("prev_lon", lag("Longitude").over(w))
df1 = df1.withColumn("prev_time", lag("timestamp_unix").over(w))

# calculating distance, time difference and speed
df1 = df1.withColumn("distance_km", haversine_udf(col("Latitude"), col("Longitude"), col("prev_lat"), col("prev_lon")))
df1 = df1.withColumn("time_diff_hr", (col("timestamp_unix") - col("prev_time")) / 3600.0)
df1 = df1.withColumn("speed_kmh", when(col("time_diff_hr") > 0, col("distance_km") / col("time_diff_hr")).otherwise(0))

#filtering suspicious speed and distance
df1 = df1.filter(
    (col("distance_km") > 0) & (col("distance_km") < 30) &    
    (col("speed_kmh") > 0) & (col("speed_kmh") < 50)         
)

# calculating all distance for that day per vessel
distance_df = df1.groupBy("MMSI").agg(_sum("distance_km").alias("total_distance_km"))
longest_vessel = distance_df.orderBy(col("total_distance_km").desc()).limit(1)

# displaying results
longest_vessel.show()







