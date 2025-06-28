import streamlit as st
import os
import shutil
import numpy as np
import cv2
from deepface import DeepFace
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import tempfile
from PIL import Image
import time

st.set_page_config(page_title="Face Clustering App", layout="wide")
st.title("🧠 Face Clustering App")
st.markdown("Cluster face images from your Google Drive folder using DeepFace + ML")

# Sidebar inputs
st.sidebar.header("Step 1: Input Settings")
image_folder = st.sidebar.text_input("🔗 Enter Google Drive image folder path")
output_folder = st.sidebar.text_input("📁 Enter Google Drive output folder path")

algorithm = st.sidebar.selectbox("📊 Choose Clustering Algorithm", ["KMeans", "DBSCAN"])

if algorithm == "KMeans":
    n_clusters = st.sidebar.number_input("Number of Clusters (K)", min_value=2, max_value=50, value=5)
elif algorithm == "DBSCAN":
    eps = st.sidebar.slider("DBSCAN eps (distance threshold)", 1.0, 50.0, 15.0)
    min_samples = st.sidebar.slider("DBSCAN min_samples", 2, 20, 5)

start_btn = st.sidebar.button("🚀 Start Clustering")

if start_btn:
    if not image_folder or not output_folder:
        st.error("Please provide both input and output folder paths.")
    else:
        st.success("Processing started...")
        image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

        embeddings = []
        valid_image_paths = []
        invalid_folder = os.path.join(output_folder, "Invalid_Faces")
        os.makedirs(invalid_folder, exist_ok=True)

        progress = st.progress(0)
        status_text = st.empty()

        for i, path in enumerate(image_paths):
            try:
                status_text.text(f"Embedding image {i+1}/{len(image_paths)}")
                reps = DeepFace.represent(img_path=path, model_name='Facenet512', detector_backend='retinaface', enforce_detection=True)

                if len(reps) > 0:
                    embeddings.append(reps[0]['embedding'])
                    valid_image_paths.append(path)
                else:
                    shutil.copy(path, invalid_folder)
            except Exception as e:
                shutil.copy(path, invalid_folder)
            progress.progress((i + 1) / len(image_paths))

        X = np.array(embeddings)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=0.90)
        X_pca = pca.fit_transform(X_scaled)

        if algorithm == "KMeans":
            model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(X_pca)
        else:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_pca)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        st.success(f"✅ Clustering complete: {n_clusters} clusters, {n_noise} noise points")

        # Save clustered images
        for label, img_path in zip(labels, valid_image_paths):
            cluster_dir = os.path.join(output_folder, f"cluster_{label}")
            os.makedirs(cluster_dir, exist_ok=True)
            shutil.copy(img_path, cluster_dir)

        st.info(f"📂 Clustered images saved to: {output_folder}")

        if st.checkbox("Show Cluster Sample Images"):
            for cluster_id in set(labels):
                if cluster_id == -1:
                    continue
                st.subheader(f"Cluster {cluster_id}")
                cluster_path = os.path.join(output_folder, f"cluster_{cluster_id}")
                cluster_imgs = os.listdir(cluster_path)[:5]  # Show first 5
                cols = st.columns(len(cluster_imgs))
                for img_file, col in zip(cluster_imgs, cols):
                    img = Image.open(os.path.join(cluster_path, img_file))
                    col.image(img, width=150)
