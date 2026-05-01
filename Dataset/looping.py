import cv2 as cv
import os
import pickle
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis()
app.prepare(ctx_id=0)

dataset_path = r"C:\Users\sumit\OneDrive\Desktop\AIML_ESE\Dataset"
database = {}
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    
    if not os.path.isdir(person_path):
        continue

    embeddings = []

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv.imread(img_path)
        if img is None:
            continue

        faces = app.get(img)

        if len(faces) == 0:
            print(f"No face in {img_name}")
            continue

        # take first face
        embedding = faces[0].embedding
        embeddings.append(embedding)

    if len(embeddings) > 0:
        database[person_name] = embeddings
        print(f"{person_name} added with {len(embeddings)} embeddings")

# save database
with open("face_db.pkl", "wb") as f:
    pickle.dump(database, f)

print("Done. Database saved.")