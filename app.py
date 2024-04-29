from flask import Flask, render_template, request, send_from_directory, url_for,redirect,flash
from werkzeug.utils import secure_filename
import os
import PyPDF2
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import string
import sqlite3
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from description_to_keyword import extract_keywords_from_job_description
import spacy
nlp = spacy.load("en_core_web_sm")
import nltk
nltk.download('stopwords')
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


app = Flask(__name__, template_folder='templates')
# Define the upload folder
UPLOAD_FOLDER = './ENGINEERING/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = Doc2Vec.load("doc2vec_resumes.model")

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = text.split()
    return tokens

def extract_skills_from_resume(resume_text):
    skills = re.findall(r'\b(?:python|sql|flask|data analysis|electrical|mechanical|automation|java|javascript|react|angular|docker|kubernetes|agile|scrum|c\+\+|c#|ruby|php|hadoop|spark|tableau|power bi|excel|machine learning|deep learning|natural language processing|computer vision|devops|git|jenkins|ansible|kubernetes|aws|azure|gcp|salesforce|sap|oracle|sql server|mysql|postgresql|mongodb|cassandra|kafka|rabbitmq|elasticsearch|kibana|grafana|prometheus|splunk|matlab|simulink|solidworks|autocad|catia|project management|kanban|waterfall|six sigma|lean|business intelligence|data warehousing|etl|data mining|data visualization|data engineering|data science|cybersecurity|network administration|system administration|database administration|software engineering|web development|mobile development|game development|embedded systems|iot|robotics|ai|ml|dl|nlp|cv)\b', resume_text, re.IGNORECASE)
    return list(set(skills))

def fetch_top_resumes(job_description):
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
        text_file_path = os.path.join("ENGINEERING", os.path.splitext(resume_file)[0] + '.txt')
        extract_text_from_pdf(os.path.join("ENGINEERING", resume_file), text_file_path)
        with open(text_file_path, 'r', encoding='utf-8') as file:
            resume_text = file.read()
        skills = extract_skills_from_resume(resume_text)
        skills_data.extend(skills)

    skill_counts = {}
    for skill in skills_data:
        skill_counts[skill] = skill_counts.get(skill, 0) + 1

    chart_data = [{'label': skill, 'value': count} for skill, count in skill_counts.items()]
    return chart_data


@app.route('/uploadResume', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return 'No file part', 400
    file = request.files['resume']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        flash('You have successfully uploaded your resume. We will contact you shortly.', 'success')
        return redirect(url_for('applyJob'))

def get_db_connection():
    conn = sqlite3.connect('dbs.db')
    return conn

def add_new_job(job_title, job_description):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        # Insert the new job into the jobs table
        c.execute("INSERT INTO jobs (job_name, job_description) VALUES (?, ?)", (job_title, job_description))
        conn.commit()
        return get_all_jobs()  # Return the updated list of jobs
    except Exception as e:
        return f"Error adding job: {e}"
    finally:
        conn.close()

 
def get_all_jobs():
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("SELECT * FROM jobs")
        jobs = [{"job_name": row[1], "job_description": row[2]} for row in c.fetchall()]
        return jobs
    except Exception as e:
        print(f"Error fetching jobs: {e}")
        return []
    finally:
        conn.close()


@app.route('/serve_resume/<filename>', methods=["GET","POST"],endpoint='serve_resume')
def serve_resume(filename):
    return send_from_directory('ENGINEERING', filename)

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/newJob")
def newJob():
    jobs = get_all_jobs()
    # Ensure jobs is always a list, even if no jobs are found
    if jobs is None:
        jobs = []
    print(jobs)
    return render_template("newJob.html", jobs=jobs)

@app.route("/applyJob")
def applyJob():
    jobs = get_all_jobs()
    # Ensure jobs is always a list, even if no jobs are found
    if jobs is None:
        jobs = []
    return render_template("applyJob.html", jobs=jobs)

@app.route("/onboard")
def onBoard():
    return render_template("onBoard.html")

@app.route("/home", methods=["POST", "GET"])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            text_file_path = os.path.splitext(file_path)[0] + '.txt'
            process_uploaded_pdf(file_path, text_file_path)
            return 'file uploaded and processed successfully'
    return render_template("home.html")

def process_uploaded_pdf(pdf_path, text_file_path):
    extract_text_from_pdf(pdf_path, text_file_path)
    with open(text_file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    cleaned_text = clean_text(text)
    print(cleaned_text)

def extract_text_from_pdf(pdf_path, text_file_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        extracted_text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            extracted_text += page_text
        with open(text_file_path, 'w', encoding='utf-8') as text_file:
            text_file.write(extracted_text)
    print(f"Text extracted from {pdf_path} and saved to {text_file_path}")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

@app.route("/addJob", methods=["POST","GET"])
def add_job():
    if request.method == 'POST':
        job_title = request.form.get('jobTitle')
        job_description = request.form.get('jobDescription')
        if job_title and job_description:
            result = add_new_job(job_title, job_description)
            return render_template("newJob.html",jobs=result)
        else:
            return "Job title and description are required."
    return render_template("/newJob")


@app.route("/result")
def result():
    job_description = request.args.get('description', '')
    top_resumes, top_resumes_with_percentages = fetch_top_resumes(job_description)
    resume_file_names = [resume_tuple[0] for resume_tuple in top_resumes]
    skills_data = extract_skills_data(resume_file_names)
    skills_data_json = json.dumps(skills_data)
    return render_template('result.html', skills_data_json=skills_data_json, top_resumes_with_percentages=top_resumes_with_percentages)

def fetch_top_resumes(job_description):
    model = Doc2Vec.load("doc2vec_resumes.model")
    keywords = extract_keywords_from_job_description(job_description)
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
    top_resumes = similarities[:10]
    top_resumes_with_percentages = [(resume, similarity * 100) for resume, similarity in top_resumes]
    return top_resumes, top_resumes_with_percentages

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = text.split()
    return tokens

def extract_skills_data(top_resumes):
    skills_data = []
    for resume_file in top_resumes:
        text_file_path = os.path.join("ENGINEERING", os.path.splitext(resume_file)[0] + '.txt')
        extract_text_from_pdf(os.path.join("ENGINEERING", resume_file), text_file_path)
        with open(text_file_path, 'r', encoding='utf-8') as file:
            resume_text = file.read()
        skills = extract_skills_from_resume(resume_text)
        skills_data.extend(skills)

    skill_counts = {}
    for skill in skills_data:
        skill_counts[skill] = skill_counts.get(skill, 0) + 1

    chart_data = [{'label': skill, 'value': count} for skill, count in skill_counts.items()]
    return chart_data

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
