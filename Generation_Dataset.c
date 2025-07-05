import cv2
import os
import json
import random
import numpy as np

def generate_dataset():
    try:
        num_users = int(input("Enter the number of users: "))
    except ValueError:
        print("Please enter a valid number.")
        return

    os.makedirs("data", exist_ok=True)
    os.makedirs("labels", exist_ok=True)

    # Load DNN face detector
    model_file = r"C:\Users\rohan\Downloads\res10_300x300_ssd_iter_140000.caffemodel"
    config_file = r"C:\Users\rohan\Downloads\deploy (1).prototxt"
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)

    def face_cropped_dnn(img):
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.9:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                startX, startY, endX, endY = box.astype("int")
                return img[startY:endY, startX:endX]
        return None

    # Load or initialize label map
    labels_file = "labels/names.json"
    if os.path.exists(labels_file):
        with open(labels_file, "r") as f:
            label_dict = json.load(f)
    else:
        label_dict = {}

    for _ in range(num_users):
        user_name = input("Enter the name of the person: ").strip()

        if user_name in label_dict.values():
            user_id = [k for k, v in label_dict.items() if v == user_name][0]
        else:
            user_id = str(len(label_dict) + 1)
            label_dict[user_id] = user_name
            with open(labels_file, "w") as f:
                json.dump(label_dict, f)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return

        print(f"Capturing images for {user_name}... Slowly move your face in all directions.")
        img_id = 0
        total_images = 500  # Increased from 300 to 500

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cropped = face_cropped_dnn(frame)
            if cropped is not None:
                img_id += 1
                face = cv2.resize(cropped, (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = cv2.equalizeHist(face)

                # Augmentation
                angle = random.randint(-15, 15)
                matrix = cv2.getRotationMatrix2D((100, 100), angle, 1)
                face_aug = cv2.warpAffine(face, matrix, (200, 200))

                filename = f"data/user.{user_id}.{img_id}.jpg"
                cv2.imwrite(filename, face_aug)
                cv2.putText(face_aug, str(img_id), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
                cv2.imshow("Capturing", face_aug)

            if cv2.waitKey(1) == 13 or img_id >= total_images:
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Collected {img_id} images for {user_name}")

generate_dataset()