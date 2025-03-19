# Import Required Packages
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
penguins_df = pd.read_csv("penguins.csv")

# Separate features and keep copy of original for output
X = penguins_df.copy()

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include='number').columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Pipeline with PCA (optional)
pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('reduce', PCA(n_components=2)),  # Optional: remove if you want to keep all features
    ('cluster', KMeans(n_clusters=4, random_state=42))
])

# Fit the pipeline
pipeline.fit(X)

# Get the transformed data before clustering (used for silhouette score)
X_transformed = pipeline.named_steps['reduce'].transform(
    pipeline.named_steps['preprocess'].transform(X)
)

# Get cluster labels
labels = pipeline.named_steps['cluster'].labels_

# Evaluate using silhouette score
sil_score = silhouette_score(X_transformed, labels)
print(f"\nSilhouette Score for k=4: {sil_score:.4f}")

# Add labels to the original DataFrame
penguins_df['Cluster'] = labels

# Final cluster summary
summary_df = penguins_df.groupby('Cluster')[
    ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
].mean()

print("\nCluster-wise summary statistics:")
print(summary_df)
