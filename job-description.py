import sqlite3
import numpy as np
import string
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from description_to_keyword import extract_keywords_from_job_description

model = Doc2Vec.load("doc2vec_resumes.model")

job_description = input("Enter the job description: ")

keywords = extract_keywords_from_job_description(job_description)
print("Extracted Keywords:")
for keyword in keywords:
    print(keyword)

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
for resume_file_name, resume_vector_bytes in resume_data:
    resume_vector = np.frombuffer(resume_vector_bytes, dtype=np.float32)
    similarity = cosine_similarity([job_description_vector], [resume_vector])[0][0]
    similarities.append((resume_file_name, similarity))

similarities.sort(key=lambda x: x[1], reverse=True)

print("\nTop 10 Resume Matches:")
for i, (resume_file_name, similarity) in enumerate(similarities[:10], start=1):
    match_percentage = similarity * 100
    print(f"{i}. {resume_file_name} - {match_percentage:.2f}%")

conn.close()