# FaceID Attendance System (AIML ESE)

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)

## Overview
The FaceID Attendance System is an automated, real-time attendance tracking solution powered by Machine Learning. It features a responsive web dashboard for student registration and monitoring, alongside a robust facial recognition pipeline that automatically logs attendance into a dual-write database (SQLite + JSON).

## Tech Stack & Algorithms
* **Face Recognition:** [InsightFace](https://github.com/deepinsight/insightface) for high-accuracy face detection and 512-D feature embedding extraction.
* **Matching Algorithm:** Cosine Similarity via `numpy` to compare live webcam embeddings against the saved database.
* **Computer Vision:** `OpenCV` for frame capturing, image decoding, and drawing UI bounding boxes.
* **Backend API:** `FastAPI` to handle HTTP requests, live SSE (Server-Sent Events) streaming, and database management.
* **Database:** `SQLite3` (Primary) and `JSON` (Mirror/Backup).
* **Frontend:** Vanilla HTML, CSS (Custom styling), and JavaScript.

## Features
- **Real-Time Recognition:** Instantly detects faces and logs attendance with high accuracy.
- **Web Dashboard:** Clean, CRT-styled UI to view live attendance logs, stats, and export data as CSV.
- **Automated Dataset Generation:** The in-browser registration tool rapidly captures 30 distinct images of a student's face in under 3 seconds to instantly build a robust training dataset.
- **Dynamic Retraining Pipeline:** The system includes a dedicated script to easily retrain the model on new datasets, updating the facial embeddings whenever a new student is registered.
- **Anti-Spam:** Session-level deduplication prevents the system from logging the same student multiple times in a single run.
