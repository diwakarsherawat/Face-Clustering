# main.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import config
from utils import load_images, extract_embeddings

# Ensure folders exist
os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)
os.makedirs(config.INVALID_FOLDER, exist_ok=True)

# Load and process images
image_paths = load_images(config.IMAGE_FOLDER)
embeddings, valid_paths = extract_embeddings(image_paths, config.INVALID_FOLDER)

# Normalize and reduce
X = np.array(embeddings)
X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=config.PCA_VARIANCE).fit_transform(X_scaled)

# KMeans
kmeans = KMeans(n_clusters=config.K_CLUSTERS, random_state=42)
labels = kmeans.fit_predict(X_pca)

# Save clustered images
for path, label in zip(valid_paths, labels):
    cluster_path = os.path.join(config.OUTPUT_FOLDER, f"Person_{label}")
    os.makedirs(cluster_path, exist_ok=True)
    shutil.copy(path, cluster_path)

# Visualize
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10')
plt.title("Face Clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()
