import pickle
import cv2 as cv
from insightface.app import FaceAnalysis
from numpy.linalg import norm
import numpy as np
import json
from datetime import datetime

with open("face_db.pkl", "rb") as f:
    database = pickle.load(f)    

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def recognize(face_embedding, database, threshold=0.5):
    best_match = "Unknown"
    best_score = -1

    for name, embeddings in database.items():
        for emb in embeddings:
            score = cosine_similarity(face_embedding, emb)

            if score > best_score:
                best_score = score
                best_match = name

    if best_score > threshold:
        return best_match
    else:
        return "Unknown"
    

app = FaceAnalysis()
app.prepare(ctx_id=0)

cap = cv.VideoCapture(0)

marked = set()  # to avoid duplicate attendance

# Load existing attendance records or create new list
attendance_file = "attendance.json"
try:
    with open(attendance_file, "r") as f:
        attendance_records = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    attendance_records = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        emb = face.embedding
        name = recognize(emb, database)

        # draw box
        x1, y1, x2, y2 = map(int, face.bbox)
        cv.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv.putText(frame, name, (x1,y1-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # attendance logic
        if name != "Unknown" and name not in marked:
            print(f"{name} marked present")
            marked.add(name)
            
            # Generate JSON record with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record = {
                "name": name,
                "timestamp": timestamp
            }
            attendance_records.append(record)
            
            # Write to JSON file
            with open(attendance_file, "w") as f:
                json.dump(attendance_records, f, indent=4)
            print(f"Record saved: {record}")

    cv.imshow("Face Recognition", frame)
    
    key = cv.waitKey(20) & 0xFF
    if key == ord('f'):
        break

cap.release()
cv.destroyAllWindows()