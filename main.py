import streamlit as st
import os
import zipfile
import shutil
import numpy as np
import uuid
from PIL import Image
from sklearn.preprocessing import StandardScaler
from deepface import DeepFace

st.set_page_config(page_title="🎯 Personalized Face Clustering", layout="wide")
st.title("📸 Face Clustering with Your Reference Photo")

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

# Uploads
st.sidebar.header("📥 Upload Files")
zip_file = st.sidebar.file_uploader("Upload ZIP of group images", type="zip")
ref_image = st.sidebar.file_uploader("Upload your reference face image", type=["jpg", "jpeg", "png"])

MAX_IMAGES = 40
RESIZE_TO = (300, 300)
DISTANCE_THRESHOLD = 10  # Adjust as needed

# Resize helper

def resize_image(path, size=(300, 300)):
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        img.thumbnail(size)
        img.save(path)
    except:
        pass

if st.sidebar.button("🚀 Start Matching"):
    if not zip_file or not ref_image:
        st.error("Please upload both the ZIP file and your reference image.")
        st.stop()

    # Extract ZIP
    with open("images.zip", "wb") as f:
        f.write(zip_file.read())
    with zipfile.ZipFile("images.zip", 'r') as zip_ref:
        zip_ref.extractall(INPUT_DIR)

    # Save reference image
    ref_path = os.path.join(REFERENCE_DIR, "ref.jpg")
    with open(ref_path, "wb") as f:
        f.write(ref_image.read())

    # Extract reference embedding
    try:
        ref_embedding = DeepFace.represent(img_path=ref_path, model_name='Facenet512', detector_backend='mtcnn')[0]['embedding']
    except:
        st.error("Could not extract reference embedding.")
        st.stop()

    # Load and process images
    image_paths = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.lower().endswith(('jpg', 'jpeg', 'png'))][:MAX_IMAGES]

    matched = 0
    unmatched = 0
    progress = st.progress(0)

    for i, path in enumerate(image_paths):
        try:
            resize_image(path, size=RESIZE_TO)
            emb = DeepFace.represent(img_path=path, model_name='Facenet512', detector_backend='mtcnn', enforce_detection=True)[0]['embedding']
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

    # Zip outputs
    shutil.make_archive("Your_Cluster", 'zip', YOUR_CLUSTER)
    shutil.make_archive("Other_Cluster", 'zip', OTHER_CLUSTER)

    # Show summary
    st.success(f"✅ Matching complete! {matched} matched, {unmatched} unmatched")

    with open("Your_Cluster.zip", "rb") as f:
        st.download_button("Download Your Photos", f, file_name="Your_Cluster.zip")

    with open("Other_Cluster.zip", "rb") as f:
        st.download_button("Download Other Cluster", f, file_name="Other_Cluster.zip")
