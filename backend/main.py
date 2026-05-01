from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import date, datetime
import sqlite3
import csv
import io
import base64
import numpy as np
import cv2
import json
import os
import asyncio

app = FastAPI(title="Attendance System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

DB_PATH = "attendance.db"
ATTENDANCE_JSON = "attendance.json"
DATASET_DIR = r"D:\Attendance AI\AIML_ESE-main\AIML_ESE-main\recognition\Dataset"


# ── Database ────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT NOT NULL,
            date      TEXT NOT NULL,
            time      TEXT NOT NULL,
            UNIQUE(name, date)
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id    INTEGER PRIMARY KEY AUTOINCREMENT,
            name  TEXT UNIQUE NOT NULL
        )
    """
    )
    conn.commit()
    conn.close()


init_db()


# ── JSON helper ──────────────────────────────────────────────
def append_to_json(name: str, timestamp: str):
    """Append a record to attendance.json (dual-write mirror)."""
    try:
        with open(ATTENDANCE_JSON, "r") as f:
            records = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        records = []
    records.append({"name": name, "timestamp": timestamp})
    with open(ATTENDANCE_JSON, "w") as f:
        json.dump(records, f, indent=4)


# ── Pydantic Models ─────────────────────────────────────────
class AttendanceIn(BaseModel):
    name: str


class StudentIn(BaseModel):
    name: str


class FaceRegisterIn(BaseModel):
    username: str
    images: list[str]


# ── Routes ──────────────────────────────────────────────────


@app.get("/", response_class=FileResponse)
def serve_dashboard():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/mark-attendance")
def mark_attendance(payload: AttendanceIn):
    """
    Called by recognizer.py when a face is recognised.
    Writes to SQLite (primary) and attendance.json (mirror).
    """
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    time_now = now.strftime("%H:%M:%S")
    timestamp = f"{today} {time_now}"

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)",
            (payload.name, today, time_now),
        )
        conn.commit()
        # Mirror to JSON
        append_to_json(payload.name, timestamp)
        return {
            "status": "marked",
            "name": payload.name,
            "date": today,
            "time": time_now,
        }
    except sqlite3.IntegrityError:
        return {"status": "already_marked", "name": payload.name}
    finally:
        conn.close()


@app.get("/attendance")
def get_attendance(date_filter: str = None):
    conn = get_db()
    if date_filter:
        rows = conn.execute(
            "SELECT * FROM attendance WHERE date = ? ORDER BY time DESC",
            (date_filter,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM attendance ORDER BY date DESC, time DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/attendance/today")
def get_today():
    today = date.today().isoformat()
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM attendance WHERE date = ? ORDER BY time DESC", (today,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/attendance/live")
async def live_attendance():
    """
    Server-Sent Events stream — pushes updated today's records every 2 s.
    The dashboard subscribes to this for zero-latency updates when a face is recognised.
    """

    async def event_generator():
        last_count = -1
        while True:
            today = date.today().isoformat()
            conn = get_db()
            rows = conn.execute(
                "SELECT * FROM attendance WHERE date = ? ORDER BY time DESC", (today,)
            ).fetchall()
            conn.close()
            count = len(rows)
            if count != last_count:
                last_count = count
                data = json.dumps([dict(r) for r in rows])
                yield f"data: {data}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/stats")
def get_stats(date_filter: str = None):
    today = date_filter or date.today().isoformat()
    conn = get_db()
    present = conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE date = ?", (today,)
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    conn.close()
    return {
        "date": today,
        "present": present,
        "total": total,
        "absent": total - present,
    }


@app.post("/register-student")
def register_student(payload: StudentIn):
    conn = get_db()
    try:
        conn.execute("INSERT INTO students (name) VALUES (?)", (payload.name,))
        conn.commit()
        return {"status": "registered", "name": payload.name}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Student already registered")
    finally:
        conn.close()


@app.get("/students")
def list_students():
    conn = get_db()
    rows = conn.execute("SELECT * FROM students ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/register_face")
def register_face(payload: FaceRegisterIn):
    username = payload.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")

    save_dir = f"{DATASET_DIR}/{username}"
    os.makedirs(save_dir, exist_ok=True)

    faces_saved = 0

    for idx, data_url in enumerate(payload.images):
        try:
            if "," in data_url:
                data_url = data_url.split(",")[1]
            img_bytes = base64.b64decode(data_url)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            # THE FIX: Save the full, colored webcam frame directly.
            # InsightFace will do the detecting later in looping.py!
            filename = f"{save_dir}/{username}_{idx}.jpg"
            cv2.imwrite(filename, frame)
            faces_saved += 1
            
        except Exception:
            continue

    if faces_saved == 0:
        raise HTTPException(
            status_code=400, detail="Failed to process captured images"
        )

    conn = get_db()
    try:
        conn.execute("INSERT INTO students (name) VALUES (?)", (username,))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()

    return {"status": "success", "username": username, "faces_saved": faces_saved}


@app.get("/export")
def export_csv(date_filter: str = None):
    conn = get_db()
    if date_filter:
        rows = conn.execute(
            "SELECT name, date, time FROM attendance WHERE date = ? ORDER BY time",
            (date_filter,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT name, date, time FROM attendance ORDER BY date, time"
        ).fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Name", "Date", "Time"])
    for row in rows:
        writer.writerow([row["name"], row["date"], row["time"]])

    output.seek(0)
    filename = f"attendance_{date_filter or 'all'}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
