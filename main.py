import streamlit as st
import os
import zipfile
import shutil
import numpy as np
import uuid
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from deepface import DeepFace

st.set_page_config(page_title="🎯 Personalized Face Clustering", layout="wide")
st.title("📸 Face Clustering with Your Reference Photo")

# Create working directories
INPUT_DIR = "temp_input"
REFERENCE_DIR = "temp_ref"
YOUR_CLUSTER = "output/Your_Cluster"
OTHER_CLUSTERS = "output/Other_Clusters"

# Clean up old runs
for folder in [INPUT_DIR, REFERENCE_DIR, YOUR_CLUSTER, OTHER_CLUSTERS]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# Upload Section
st.sidebar.header("📥 Upload Files")
zip_file = st.sidebar.file_uploader("Upload ZIP of group images", type="zip")
ref_image = st.sidebar.file_uploader("Upload your reference face image", type=["jpg", "jpeg", "png"])

MAX_IMAGES = 40
RESIZE_TO = (300, 300)

# Resize helper
def resize_image(path, size=(300, 300)):
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        img.thumbnail(size)
        img.save(path)
    except Exception as e:
        pass

if st.sidebar.button("🚀 Start Clustering"):
    if not zip_file or not ref_image:
        st.error("Please upload both the ZIP file and your reference image.")
    else:
        # Save and extract ZIP
        with open("images.zip", "wb") as f:
            f.write(zip_file.read())

        with zipfile.ZipFile("images.zip", 'r') as zip_ref:
            zip_ref.extractall(INPUT_DIR)

        # Save reference image
        ref_path = os.path.join(REFERENCE_DIR, "ref.jpg")
        with open(ref_path, "wb") as f:
            f.write(ref_image.read())

        # Load all valid image paths (limit to MAX_IMAGES)
        image_paths = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        image_paths = image_paths[:MAX_IMAGES]

        embeddings = []
        valid_paths = []

        st.subheader("🔎 Extracting Embeddings")
        progress = st.progress(0)
        for i, path in enumerate(image_paths):
            try:
                resize_image(path, size=RESIZE_TO)
                rep = DeepFace.represent(img_path=path, model_name='Facenet512', detector_backend='mtcnn', enforce_detection=True)
                if rep:
                    embeddings.append(rep[0]['embedding'])
                    valid_paths.append(path)
            except:
                continue
            progress.progress((i+1)/len(image_paths))

        if len(embeddings) < 2:
            st.error("Not enough valid faces found to perform clustering.")
            st.stop()

        # Preprocess embeddings
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(embeddings)
        pca = PCA(n_components=0.90)
        X_pca = pca.fit_transform(X_scaled)

        st.subheader("📊 Clustering Faces")
        kmeans = KMeans(n_clusters=5, random_state=42)
        labels = kmeans.fit_predict(X_pca)

        # Embed reference image
        st.subheader("📌 Matching Your Reference Face")
        try:
            ref_embedding = DeepFace.represent(img_path=ref_path, model_name='Facenet512', detector_backend='mtcnn')[0]['embedding']
            ref_scaled = scaler.transform([ref_embedding])
            ref_pca = pca.transform(ref_scaled)
            ref_cluster = kmeans.predict(ref_pca)[0]
        except:
            st.error("Could not process reference face.")
            st.stop()

        # Separate clusters
        for label, path in zip(labels, valid_paths):
            if label == ref_cluster:
                cluster_dir = os.path.join(YOUR_CLUSTER)
            else:
                cluster_dir = os.path.join(OTHER_CLUSTERS, f"cluster_{label}")
            os.makedirs(cluster_dir, exist_ok=True)
            shutil.copy(path, os.path.join(cluster_dir, os.path.basename(path)))

        # Zip results
        shutil.make_archive("Your_Cluster", 'zip', YOUR_CLUSTER)
        shutil.make_archive("Other_Clusters", 'zip', OTHER_CLUSTERS)

        # Download buttons
        st.subheader("📁 Download Results")
        with open("Your_Cluster.zip", "rb") as f:
            st.download_button("Download Your Photos", f, file_name="Your_Cluster.zip")

        with open("Other_Clusters.zip", "rb") as f:
            st.download_button("Download Other Clusters", f, file_name="Other_Clusters.zip")

        st.success("✅ Clustering complete!")
