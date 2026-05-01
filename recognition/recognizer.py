import pickle
import cv2 as cv
from insightface.app import FaceAnalysis
from numpy.linalg import norm
import numpy as np
import json
import requests
from datetime import datetime

# ── Load face database ───────────────────────────────────────
with open("face_db.pkl", "rb") as f:
    database = pickle.load(f)

FASTAPI_URL = "http://localhost:8000"
ATTENDANCE_JSON = "attendance.json"

# ── Load existing JSON attendance records ────────────────────
try:
    with open(ATTENDANCE_JSON, "r") as f:
        attendance_records = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    attendance_records = []


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
    return best_match if best_score > threshold else "Unknown"


def mark_attendance(name: str):
    """
    Dual-write: append to attendance.json AND POST to FastAPI (which writes to SQLite).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {"name": name, "timestamp": timestamp}

    # 1️⃣  Write to JSON
    attendance_records.append(record)
    with open(ATTENDANCE_JSON, "w") as f:
        json.dump(attendance_records, f, indent=4)

    # 2️⃣  Write to SQLite via FastAPI
    try:
        resp = requests.post(
            f"{FASTAPI_URL}/mark-attendance",
            json={"name": name},
            timeout=2,
        )
        data = resp.json()
        if data.get("status") == "already_marked":
            print(f"[INFO] {name} already marked today (DB).")
        else:
            print(f"[OK] {name} marked present at {timestamp}")
    except requests.exceptions.RequestException as e:
        print(f"[WARN] Could not reach FastAPI: {e}  (JSON record still saved)")


# ── InsightFace setup ────────────────────────────────────────
app = FaceAnalysis()
app.prepare(ctx_id=0)          # ctx_id=-1 for CPU only

cap = cv.VideoCapture(0)
marked = set()                 # session-level dedup (one mark per run)

print("[INFO] Face recognition started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        emb = face.embedding
        name = recognize(emb, database)

        x1, y1, x2, y2 = map(int, face.bbox)
        color = (0, 220, 100) if name != "Unknown" else (0, 80, 220)
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv.putText(frame, name, (x1, y1 - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Mark attendance once per session
        if name != "Unknown" and name not in marked:
            marked.add(name)
            mark_attendance(name)

    cv.imshow("Face Recognition — press Q to quit", frame)
    if cv.waitKey(20) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
print("[INFO] Recognition session ended.")