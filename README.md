# Face-Clustering

# 📸 Face-Based Photo Clustering Project

## 🧠 Problem
When traveling with friends/family, we often take hundreds of pictures across different devices. After the trip, people share photos through Drive or other cloud platforms. But finding *your own photos* is tedious and manual.

## ✅ Solution
This project automatically clusters similar faces from a folder of photos and organizes them into folders by person.

## 💡 Features
- Face detection using RetinaFace
- Face embedding using Facenet512
- PCA for dimensionality reduction (90% variance)
- KMeans clustering to group similar faces
- Separate folder for blurry or undetectable faces
- Designed to run on **Google Colab + Google Drive**

## 📂 Folder Output
- `So/Person_0`, `So/Person_1`, ... → Clustered folders for each person
- `Invalid_Faces/` → Undetectable or blurry face photos

## 🚀 Run the Project
1. Open `main.ipynb` in Google Colab
2. Mount your Drive
3. Upload your images to a folder in Drive
4. Run the notebook
