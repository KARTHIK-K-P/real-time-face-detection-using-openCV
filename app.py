import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime, timedelta

# === Configuration ===
known_faces_path = r"face recognition\data_setofatt"
recognized_faces_path = r'encodings\recognized_faces'
unknown_faces_path = r'encodings\unknown_faces'

# === Ensure directories exist ===
os.makedirs(recognized_faces_path, exist_ok=True)
os.makedirs(unknown_faces_path, exist_ok=True)

# === Load known faces ===
images = []
classNames = []

print("Loading known faces...")
for filename in os.listdir(known_faces_path):
    img_path = os.path.join(known_faces_path, filename)
    img = cv2.imread(img_path)
    if img is not None:
        images.append(img)
        classNames.append(os.path.splitext(filename)[0])
    else:
        print(f"Warning: Couldn't load image {filename}")

def findEncodings(images):
    encodeList = []
    for img in images:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_img)
        if encodings:
            encodeList.append(encodings[0])
        else:
            print("No face found in one of the images.")
    return encodeList

encodeListKnown = findEncodings(images)
print(f"Encoding complete. Found {len(encodeListKnown)} known faces.")

# === Camera and recognition setup ===
cap = cv2.VideoCapture(0)
recognized_faces = set()
last_unknown_save_time = datetime.min

# === Create a timestamped CSV file ===
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
csv_file_name = f'Attendance_{timestamp}.csv'
with open(csv_file_name, 'w') as f:
    f.write('Name,Time\n')

def markAttendance(name):
    now = datetime.now().strftime('%d:%m:%Y  %H:%M:%S')
    df = pd.read_csv(csv_file_name)
    if name not in df['Name'].values:
        with open(csv_file_name, 'a') as f:
            f.write(f'{name},{now}\n')

print("Starting camera. Press 'q' to quit...")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from webcam.")
        break

    small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    rgb_small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

    faces_in_frame = face_recognition.face_locations(rgb_small_img)
    encodes_in_frame = face_recognition.face_encodings(rgb_small_img, faces_in_frame)

    for encodeFace, faceLoc in zip(encodes_in_frame, faces_in_frame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis) if faceDis.size > 0 else -1

        y1, x2, y2, x1 = [v * 4 for v in faceLoc]

        if matchIndex != -1 and matches[matchIndex] and faceDis[matchIndex] < 0.5:
            name = classNames[matchIndex].upper()
            if name not in recognized_faces:
                recognized_faces.add(name)
                markAttendance(name)
                face_img_path = os.path.join(recognized_faces_path, f'{name}_{timestamp}.jpg')
                cv2.imwrite(face_img_path, img[y1:y2, x1:x2])
            color = (0, 255, 0)
            label = name
        else:
            if (datetime.now() - last_unknown_save_time) > timedelta(seconds=30):
                unknown_face_path = os.path.join(unknown_faces_path, f'unknown_{timestamp}.jpg')
                cv2.imwrite(unknown_face_path, img[y1:y2, x1:x2])
                last_unknown_save_time = datetime.now()
            color = (0, 0, 255)
            label = 'Unknown'

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
        cv2.putText(img, label, (x1 + 6, y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Webcam - Press q to Quit', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Attendance saved in {csv_file_name}")
