# Customer Purchase Pattern Analysis

This project uses a Spark-based pipeline to process retail transaction data and perform RFM-based customer segmentation using K-Means clustering.

## Features
- Spark-powered ETL pipeline for large-scale retail data
- RFM (Recency, Frequency, Monetary) feature engineering
- K-Means clustering with Elbow and Silhouette analysis
- Cluster visualization with Seaborn

## Getting Started
```bash
pip install -r requirements.txt
python rfm_clustering.py
```

## Output
- `rfm_clustered_output.csv` — clustered customer data
- `elbow_plot.png` — elbow chart
- `cluster_plot.png` — segmentation plot
