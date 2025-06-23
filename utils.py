# utils.py

import os
import shutil
from deepface import DeepFace

def load_images(image_folder):
    return [os.path.join(image_folder, f) for f in os.listdir(image_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

def extract_embeddings(image_paths, invalid_folder):
    embeddings = []
    valid_paths = []

    for path in image_paths:
        try:
            reps = DeepFace.represent(img_path=path,
                                      model_name='Facenet512',
                                      detector_backend='retinaface',
                                      enforce_detection=True)
            if len(reps) > 0:
                embeddings.append(reps[0]['embedding'])
                valid_paths.append(path)
            else:
                shutil.copy(path, invalid_folder)
        except Exception:
            shutil.copy(path, invalid_folder)

    return embeddings, valid_paths
