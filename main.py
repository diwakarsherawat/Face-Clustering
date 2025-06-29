import streamlit as st
import os
import shutil
import numpy as np
import time
import requests
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logging.info("🟢 App started or restarted")

st.set_page_config(page_title="Face Match Clustering", layout="centered")
st.markdown("""
    <h1 style='text-align: center; color: #2c3e50;'>🔍 Face Match Clustering</h1>
    <h4 style='text-align: center; color: #7f8c8d;'>Upload up to 100 face images and one reference image. We'll identify which ones belong to you.</h4>
""", unsafe_allow_html=True)

# Define folders
TEMP_DIR = "temp_input"
REFERENCE_DIR = "temp_ref"
for folder in [TEMP_DIR, REFERENCE_DIR]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# Sidebar: File Uploads
st.sidebar.header("📥 Upload Images")
group_images = st.sidebar.file_uploader("Upload multiple group images (max 100)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
ref_image = st.sidebar.file_uploader("Upload your reference face image", type=["jpg", "jpeg", "png"])

model_name = st.sidebar.selectbox("🧠 Choose Face Recognition Model", ["SFace", "Facenet", "Facenet512", "VGG-Face"])
clustering_algorithm = st.sidebar.selectbox("📊 Choose Clustering Algorithm", ["KMeans", "DBSCAN"])
if clustering_algorithm == "KMeans":
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 20, 5)
elif clustering_algorithm == "DBSCAN":
    eps = st.sidebar.slider("DBSCAN eps", 5.0, 50.0, 15.0)
    min_samples = st.sidebar.slider("min_samples", 2, 10, 3)

MAX_IMAGES = 100
RESIZE_SCALE = 0.75

API_URL = "https://26dd-35-185-13-139.ngrok-free.app/cluster"
API_KEY = os.environ.get("API_KEY")

def resize_image_percent(path, scale=0.75):
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        w, h = img.size
        img = img.resize((int(w * scale), int(h * scale)))
        img.save(path)
    except:
        pass

if st.sidebar.button("🚀 Start Matching"):
    if not group_images or not ref_image:
        st.error("Please upload at least one reference image and group images.")
        st.stop()

    st.info("📡 Sending images to backend API...")

    # Save reference image to list of files
    ref_path = os.path.join(REFERENCE_DIR, "ref.jpg")
    with open(ref_path, "wb") as f:
        f.write(ref_image.read())

    files = [("images", ("ref.jpg", open(ref_path, "rb"), "image/jpeg"))]
    for img in group_images[:MAX_IMAGES]:
        files.append(("images", (img.name, img, "image/jpeg")))

    # Prepare clustering options
    data = {
        "model_name": model_name,
        "algorithm": clustering_algorithm,
        "n_clusters": str(n_clusters) if clustering_algorithm == "KMeans" else "",
        "eps": str(eps) if clustering_algorithm == "DBSCAN" else "",
        "min_samples": str(min_samples) if clustering_algorithm == "DBSCAN" else ""
    }

    headers = {"x-api-key": API_KEY}

    try:
        response = requests.post(API_URL, files=files, data=data, headers=headers)
        if response.status_code == 200:
            st.success("✅ Matching complete! Download your matched and unmatched clusters:")
            st.download_button("⬇️ Download Clustered ZIP", response.content, file_name="clusters.zip")
        else:
            st.error(f"❌ Error from backend: {response.text}")
    except Exception as e:
        st.error(f"🚨 Request failed: {e}")

    st.info("💡 Tip: Upload images where the faces are clear and well-lit for better accuracy.")
