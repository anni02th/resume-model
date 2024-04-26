import pdfplumber
import string
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import sqlite3
import numpy as np

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = text.split()
    return tokens

def train_doc2vec(documents, vector_size=50, window=5, min_count=5, workers=8, epochs=100):
    tagged_documents = [TaggedDocument(words=preprocess_text(doc), tags=[str(i)]) for i, doc in enumerate(documents)]
    model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers, dm=1)
    model.build_vocab(tagged_documents)
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha
    model.save("doc2vec_resumes.model")
    return model

pdf_directory = "ENGINEERING"
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

documents = []

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    text = extract_text_from_pdf(pdf_path)
    documents.append(text)

if not os.path.exists("doc2vec_resumes.model"):
    model = train_doc2vec(documents)
else:
    model = Doc2Vec.load("doc2vec_resumes.model")

# Vectorize resumes
resume_vectors = [model.infer_vector(preprocess_text(doc)) for doc in documents]

conn = sqlite3.connect('resume_vectors.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS resumes
             (pdf_file_name TEXT PRIMARY KEY, vector BLOB)''')

for i, vector in enumerate(resume_vectors):
    try:
        vector_bytes = np.array(vector).tobytes()
        c.execute("INSERT INTO resumes (pdf_file_name, vector) VALUES (?, ?)", (pdf_files[i], vector_bytes))
    except Exception as e:
        print(f"Error inserting vector {i}: {e}")


conn.commit()
conn.close()