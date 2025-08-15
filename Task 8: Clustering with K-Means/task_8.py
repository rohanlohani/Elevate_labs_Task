# task_8.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. Load dataset
df = pd.read_csv("/home/rohan/Desktop/Elevate Labs/Task 8: Clustering with K-Means/customer_segmentation.csv", encoding="latin1")

# 2. Clean data: remove rows without CustomerID
df = df.dropna(subset=['CustomerID'])

# 3. Aggregate data per customer
customer_data = df.groupby('CustomerID').agg({
    'InvoiceNo': 'nunique',   # Number of transactions
    'Quantity': 'sum',        # Total quantity purchased
    'UnitPrice': 'mean',      # Average price per unit
}).rename(columns={
    'InvoiceNo': 'NumTransactions',
    'Quantity': 'TotalQuantity',
    'UnitPrice': 'AvgUnitPrice'
}).reset_index()

# Features for clustering
features = customer_data[['NumTransactions', 'TotalQuantity', 'AvgUnitPrice']]

# 4. Normalize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 5. Elbow Method to find optimal k
inertia = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# 6. Choose optimal k based on elbow (change this if needed)
optimal_k = 4

# 7. Fit K-Means with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customer_data['Cluster'] = kmeans.fit_predict(scaled_features)

# 8. Evaluate with Silhouette Score
sil_score = silhouette_score(scaled_features, customer_data['Cluster'])
print(f"Silhouette Score: {sil_score:.4f}")

# 9. PCA for 2D visualization
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

plt.figure(figsize=(6, 4))
for cluster in range(optimal_k):
    plt.scatter(pca_features[customer_data['Cluster'] == cluster, 0],
                pca_features[customer_data['Cluster'] == cluster, 1],
                label=f'Cluster {cluster}')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Customer Segments (PCA)')
plt.legend()
plt.show()

# 10. Save results
customer_data.to_csv("clustered_customers.csv", index=False)
print("Clustered customer data saved to clustered_customers.csv")
