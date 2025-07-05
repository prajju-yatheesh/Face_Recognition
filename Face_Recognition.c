import cv2
import numpy as np
import json
import time
import winsound

# Load DNN face detector model
model_file = r"C:\Users\rohan\Downloads\res10_300x300_ssd_iter_140000.caffemodel"
config_file = r"C:\Users\rohan\Downloads\deploy (1).prototxt"
net = cv2.dnn.readNetFromCaffe(config_file, model_file)

# Load name mappings
with open("labels/names.json", "r") as f:
    names = json.load(f)

# Load LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("classifier.xml")

# Path to loud siren sound (.wav file)
siren_path = r"C:\Users\rohan\Downloads\Warning Siren-SoundBible.com-898272278 (1).wav"

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

print("🔍 Press 'Enter' to exit...")

# Timer and alert state tracking
unknown_start_time = None
alert_active = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    unknown_detected = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            face = frame[y1:y2, x1:x2]

            try:
                gray_face = cv2.cvtColor(cv2.resize(face, (200, 200)), cv2.COLOR_BGR2GRAY)
                id, pred = recognizer.predict(gray_face)

                # Confidence calculation
                conf = int(100 * (1 - pred / 100))
                print(f"[INFO] ID: {id}, Prediction Score: {pred:.2f}, Confidence: {conf}%")

                if conf > 85:
                    name = names.get(str(id), f"User {id}")
                    color = (0, 255, 0)  # Green
                else:
                    name = "UNKNOWN"
                    unknown_detected = True
                    color = (0, 0, 255)  # Red

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} ({conf}%)", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            except Exception as e:
                print(f"Error processing face: {e}")

    # Timer logic for unknown face
    current_time = time.time()
    if unknown_detected:
        if unknown_start_time is None:
            unknown_start_time = current_time
        elif current_time - unknown_start_time > 3:
            if not alert_active:
                print("⚠ UNKNOWN person detected for over 3 seconds! Playing siren.")
                alert_active = True
                winsound.PlaySound(siren_path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
    else:
        unknown_start_time = None
        if alert_active:
            print("✅ UNKNOWN person left. Stopping siren.")
            winsound.PlaySound(None, winsound.SND_PURGE)
            alert_active = False

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == 13:  # Enter key
        break

cap.release()
cv2.destroyAllWindows()
winsound.PlaySound(None, winsound.SND_PURGE)  # Stop any remaining sound