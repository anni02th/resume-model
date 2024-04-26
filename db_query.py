import string
from gensim.models.doc2vec import Doc2Vec
import sqlite3
import numpy as np

model = Doc2Vec.load("doc2vec_resumes.model")

job_description = "AutoCAD"

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = text.split()
    return tokens

job_description_tokens = preprocess_text(job_description)
job_description_vector = model.infer_vector(job_description_tokens)

conn = sqlite3.connect('resume_vectors.db')
c = conn.cursor()

c.execute("SELECT pdf_file_name, vector FROM resumes")
resume_data = c.fetchall()

similarities = []
for resume_id, resume_vector_bytes in resume_data:
    resume_vector = np.frombuffer(resume_vector_bytes, dtype=np.float32)
    similarity = np.dot(resume_vector, job_description_vector) / (np.linalg.norm(resume_vector) * np.linalg.norm(job_description_vector))
    similarities.append((resume_id, similarity))

similarities.sort(key=lambda x: x[1], reverse=True)

top_pdf_file_names = [x[0] for x in similarities[:5]]

print("Top matching resumes for ", job_description, ": ")
for pdf_file_name in top_pdf_file_names:
    print(f"PDF File Name: {pdf_file_name}")
