import streamlit as st
import os
import shutil
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from deepface import DeepFace
import time
import logging

logging.basicConfig(level=logging.INFO)
logging.info("🟢 App started or restarted")

st.set_page_config(page_title="Face Match Clustering", layout="centered")
st.markdown("""
    <h1 style='text-align: center; color: #2c3e50;'>🔍 Face Match Clustering</h1>
    <h4 style='text-align: center; color: #7f8c8d;'>Upload up to 100 face images and one reference image. We'll identify which ones belong to you.</h4>
""", unsafe_allow_html=True)

# Define folders
INPUT_DIR = "temp_input"
REFERENCE_DIR = "temp_ref"
YOUR_CLUSTER = "output/Your_Cluster"
OTHER_CLUSTER = "output/Other_Cluster"

# Clear previous data
for folder in [INPUT_DIR, REFERENCE_DIR, YOUR_CLUSTER, OTHER_CLUSTER]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# Sidebar: File Uploads
st.sidebar.header("📥 Upload Images")
group_images = st.sidebar.file_uploader("Upload multiple group images (max 100)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
ref_image = st.sidebar.file_uploader("Upload your reference face image", type=["jpg", "jpeg", "png"])

MAX_IMAGES = 100
DISTANCE_THRESHOLD = 0.65  # Optimized for SFace
RESIZE_SCALE = 0.75

# Resize helper with % scale
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

    st.info("📦 Preprocessing images and extracting embeddings...")

    # Save reference image
    ref_path = os.path.join(REFERENCE_DIR, "ref.jpg")
    with open(ref_path, "wb") as f:
        f.write(ref_image.read())

    try:
        with st.spinner("⚙️ Loading SFace model... This may take up to 30 seconds."):
            ref_embedding = DeepFace.represent(img_path=ref_path, model_name='SFace', detector_backend='opencv', enforce_detection=False)[0]['embedding']
    except:
        st.error("Could not extract reference embedding.")
        st.stop()

    # Save group images
    uploaded_images = group_images[:MAX_IMAGES]
    image_paths = []
    for img_file in uploaded_images:
        filename = os.path.join(INPUT_DIR, img_file.name)
        with open(filename, "wb") as f:
            f.write(img_file.read())
        resize_image_percent(filename, scale=RESIZE_SCALE)
        image_paths.append(filename)

    # Process and match
    matched = 0
    unmatched = 0
    progress = st.progress(0)
    status_text = st.empty()

    for i, path in enumerate(image_paths):
        try:
            emb = DeepFace.represent(img_path=path, model_name='SFace', detector_backend='opencv', enforce_detection=False)[0]['embedding']
            distance = np.linalg.norm(np.array(ref_embedding) - np.array(emb))
            if distance < DISTANCE_THRESHOLD:
                shutil.copy(path, os.path.join(YOUR_CLUSTER, os.path.basename(path)))
                matched += 1
            else:
                shutil.copy(path, os.path.join(OTHER_CLUSTER, os.path.basename(path)))
                unmatched += 1
        except:
            continue

        progress.progress((i+1)/len(image_paths))
        status_text.text(f"🔄 Processing {i+1} of {len(image_paths)} images")
        time.sleep(0.2)

    if matched == 0 and unmatched == 0:
        st.warning("⚠️ No faces processed. Please check image quality or supported formats.")
        st.stop()

    # Zip outputs
    shutil.make_archive("Your_Cluster", 'zip', YOUR_CLUSTER)
    shutil.make_archive("Other_Cluster", 'zip', OTHER_CLUSTER)

    st.success(f"✅ Matching complete! {matched} matched, {unmatched} unmatched")

    col1, col2 = st.columns(2)
    with col1:
        with open("Your_Cluster.zip", "rb") as f:
            st.download_button("⬇️ Download Your Matches", f, file_name="Your_Cluster.zip")
    with col2:
        with open("Other_Cluster.zip", "rb") as f:
            st.download_button("⬇️ Download Other Cluster", f, file_name="Other_Cluster.zip")

    st.info("💡 Tip: Upload images where the faces are clear and well-lit for better accuracy.")
