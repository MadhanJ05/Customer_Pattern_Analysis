from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max as spark_max, min as spark_min, count, sum as spark_sum, datediff, current_date
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

spark = SparkSession.builder \
    .appName("Customer Purchase Pattern Analysis") \
    .getOrCreate()

df = spark.read.csv("data/transactions.csv", header=True, inferSchema=True)

rfm = df.groupBy("CustomerID").agg(
    datediff(current_date(), spark_max("InvoiceDate")).alias("Recency"),
    count("InvoiceNo").alias("Frequency"),
    spark_sum("TotalAmount").alias("Monetary")
)

rfm_pd = rfm.toPandas()
rfm_pd.dropna(inplace=True)

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_pd[["Recency", "Frequency", "Monetary"]])

inertia = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(rfm_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.savefig("elbow_plot.png")
plt.close()

best_k = 4
kmeans = KMeans(n_clusters=best_k, random_state=42)
rfm_pd['Cluster'] = kmeans.fit_predict(rfm_scaled)
score = silhouette_score(rfm_scaled, rfm_pd['Cluster'])
print(f"Silhouette Score for {best_k} clusters: {score:.2f}")

plt.figure(figsize=(8,6))
sns.scatterplot(data=rfm_pd, x="Recency", y="Monetary", hue="Cluster", palette="tab10")
plt.title("Customer Segmentation (Recency vs. Monetary)")
plt.savefig("cluster_plot.png")
plt.close()

rfm_pd.to_csv("rfm_clustered_output.csv", index=False)
print("✅ RFM clustering completed and exported.")
spark.stop()
