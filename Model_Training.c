import os
import cv2
from PIL import Image
import numpy as np
import time

def train_classifier(data_dir):
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    faces = []
    ids = []

    for idx, image in enumerate(image_paths):
        try:
            img = Image.open(image).convert('L')
            image_np = np.array(img, 'uint8')
            image_np = cv2.equalizeHist(image_np)
            image_np = cv2.resize(image_np, (100, 100))  # Smaller size = faster
            id = int(os.path.split(image)[1].split(".")[1])
            faces.append(image_np)
            ids.append(id)
        except Exception as e:
            print(f"Skipping {image}: {e}")

    ids = np.array(ids)

    clf = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=4,
        grid_y=4
    )

    print("⏳ Starting training...")
    start_time = time.time()
    clf.train(faces, ids)
    duration = time.time() - start_time
    print(f"✅ Training completed in {duration:.2f} seconds.")

    clf.write("classifier.xml")

train_classifier("data")