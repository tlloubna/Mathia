import os
import sys
from pathlib import Path

# Add parent directory to path
extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)



import json 

import json

file_json = "/home/loubna/Code_Projet_Mathia/Mathia/data/Mathiadata2/learninglocker_v2.statements_mars2026.json"
output_json = "/home/loubna/Code_Projet_Mathia/Mathia/data/Mathiadata2/data_small.json"

MAX_STUDENTS = 50  # nombre d'élèves à garder

students_seen = set()
kept = []

with open(file_json, "r", encoding="utf-8") as f:
    for line in f:
        try:
            doc = json.loads(line)
            student_id = doc["statement"]["actor"]["account"]["name"]
            
            if student_id not in students_seen:
                if len(students_seen) >= MAX_STUDENTS:
                    continue
                students_seen.add(student_id)
            
            kept.append(doc)
        except:
            continue

with open(output_json, "w", encoding="utf-8") as f:
    for doc in kept:
        f.write(json.dumps(doc) + "\n")

print(f"Original : {len(kept)} lignes gardées")
print(f"Élèves : {len(students_seen)}")

