import json
import os
from flask import Flask, render_template
import sqlite3
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import re, string

from description_to_keyword import extract_keywords_from_job_description
from vectorized_resume import extract_text_from_pdf

app = Flask(__name__)

model = Doc2Vec.load("doc2vec_resumes.model")

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = text.split()
    return tokens

def extract_skills_from_resume(resume_text):
    skills = re.findall(r'\b(?:python|sql|flask|data analysis|electrical|mechanical|automation|java|javascript|react|angular|docker|kubernetes|agile|scrum|c\+\+|c#|ruby|php|hadoop|spark|tableau|power bi|excel|machine learning|deep learning|natural language processing|computer vision|devops|git|jenkins|ansible|kubernetes|aws|azure|gcp|salesforce|sap|oracle|sql server|mysql|postgresql|mongodb|cassandra|kafka|rabbitmq|elasticsearch|kibana|grafana|prometheus|splunk|matlab|simulink|solidworks|autocad|catia|project management|kanban|waterfall|six sigma|lean|business intelligence|data warehousing|etl|data mining|data visualization|data engineering|data science|cybersecurity|network administration|system administration|database administration|software engineering|web development|mobile development|game development|embedded systems|iot|robotics|ai|ml|dl|nlp|cv)\b', resume_text, re.IGNORECASE)
    return list(set(skills))

def fetch_top_resumes():
    job_description = input("Enter the job description: ")
    keywords = extract_keywords_from_job_description(job_description)
    print("Extracted Keywords:")
    for keyword in keywords:
        print(keyword)

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

    top_resumes = [resume for resume, _ in similarities[:10]]

    conn.close()
    return top_resumes

def extract_skills_data(top_resumes):
    skills_data = []
    for resume_file in top_resumes:
        resume_text = extract_text_from_pdf(os.path.join("ENGINEERING", resume_file))
        skills = extract_skills_from_resume(resume_text)
        skills_data.extend(skills)

    skill_counts = {}
    for skill in skills_data:
        skill_counts[skill] = skill_counts.get(skill, 0) + 1

    chart_data = [{'label': skill, 'value': count} for skill, count in skill_counts.items()]
    return chart_data


@app.route('/')
def index():
    top_resumes = fetch_top_resumes()
    skills_data = extract_skills_data(top_resumes)
    skills_data_json = json.dumps(skills_data)
    return render_template('result.html', skills_data_json=skills_data_json)

if __name__ == '__main__':
    app.run(debug=True)