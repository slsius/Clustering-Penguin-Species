# Import Required Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
penguins_df = pd.read_csv("penguins.csv")
penguins_dummies = pd.get_dummies(penguins_df)

# Scale the data
scaler = StandardScaler()
penguins_scaled = scaler.fit_transform(penguins_dummies)

pca = PCA(n_components=2)
penguins_pca = pca.fit_transform(penguins_scaled)

# Elbow plot to determine best k
distortions = []
num_clusters = range(1, 11)

for i in num_clusters:
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(penguins_scaled)
    distortions.append(kmeans.inertia_)

# Plot elbow curve
elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})
sns.lineplot(x='num_clusters', y='distortions', data=elbow_plot)
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters")
plt.ylabel("Distortion (Inertia)")
plt.xticks(num_clusters)
plt.grid(True)
plt.show()

# Assume best k = 4 (based on elbow plot)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(penguins_scaled)
labels = kmeans.labels_

# Add labels to the original DataFrame
#penguins_scaled['label'] = labels
penguins_df['label'] = labels

# Plot two features colored by cluster (e.g., culmen_length_mm vs flipper_length_mm)
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=penguins_df['culmen_length_mm'], 
    y=penguins_df['flipper_length_mm'], 
    hue=penguins_df['label'], 
    palette='Set1'
)
plt.title("KMeans Clustering (k=4)")
plt.xlabel("Culmen Length (mm)")
plt.ylabel("Flipper Length (mm)")
plt.legend(title="Cluster")
plt.show()

# Plot two features colored by cluster (e.g., culmen_depth_mm vs flipper_length_mm)
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=penguins_df['culmen_depth_mm'], 
    y=penguins_df['flipper_length_mm'], 
    hue=penguins_df['label'], 
    palette='Set1'
)
plt.title("KMeans Clustering (k=4)")
plt.xlabel("Culmen depth (mm)")
plt.ylabel("Flipper Length (mm)")
plt.legend(title="Cluster")
plt.show()

# Plot two features colored by cluster (e.g., culmen_depth_mm vs culmen_length_mm)
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=penguins_df['culmen_depth_mm'], 
    y=penguins_df['culmen_length_mm'], 
    hue=penguins_df['label'], 
    palette='Set1'
)
plt.title("KMeans Clustering (k=4)")
plt.xlabel("Culmen depth (mm)")
plt.ylabel("Flipper Length (mm)")
plt.legend(title="Cluster")
plt.show()

#pca graph
pca_df = pd.DataFrame(data=penguins_pca, columns=['PC1', 'PC2'])
pca_df['Cluster'] = labels

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='Set2')
plt.title("KMeans Clustering Visualized with PCA (k=4)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()


#Final Result
numeric_columns = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm','label']
stat_penguins = penguins_df[numeric_columns].groupby('label').mean()
stat_penguins
