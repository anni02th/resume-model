import spacy
from spacy.matcher import Matcher
import yake
import re
from collections import Counter

nlp = spacy.load("en_core_web_sm")

kw_extractor = yake.KeywordExtractor()

# Define an expanded dictionary of relevant keywords
keyword_dict = {
    "skills": ["python", "sql", "flask", "data analysis", "electrical", "mechanical", "automation", "java", "javascript", "react", "angular", "docker", "kubernetes", "agile", "scrum", "c++", "c#", "ruby", "php", "hadoop", "spark", "tableau", "power bi", "excel", "tableau", "machine learning", "deep learning", "natural language processing", "computer vision", "devops", "git", "jenkins", "ansible", "kubernetes", "aws", "azure", "gcp", "salesforce", "sap", "oracle", "sql server", "mysql", "postgresql", "mongodb", "cassandra", "kafka", "rabbitmq", "elasticsearch", "kibana", "grafana", "prometheus", "splunk", "tableau", "power bi", "excel", "matlab", "simulink", "solidworks", "autocad", "catia", "solidworks", "project management", "agile", "scrum", "kanban", "waterfall", "six sigma", "lean", "business intelligence", "data warehousing", "etl", "data mining", "data visualization", "data engineering", "data science", "cybersecurity", "network administration", "system administration", "database administration", "software engineering", "web development", "mobile development", "game development", "embedded systems", "iot", "robotics", "ai", "ml", "dl", "nlp", "cv"],
    "certifications": ["hvac", "electrical", "welding", "pmp", "cissp", "aws", "azure", "cisco", "oracle", "google cloud", "comptia a+", "comptia network+", "comptia security+", "comptia linux+", "comptia project+", "itil", "prince2", "capm", "pmp", "cisa", "cism", "cissp", "ccna", "ccnp", "ccie", "mcsa", "mcse", "mcsd", "aws certified solutions architect", "aws certified developer", "aws certified sysops administrator", "azure certified administrator", "azure certified developer", "azure certified solutions architect", "gcp certified associate cloud engineer", "gcp certified professional cloud architect", "salesforce certified administrator", "salesforce certified developer", "salesforce certified consultant", "sap certified application associate", "sap certified technology associate", "oracle certified associate", "oracle certified professional", "mysql certified developer", "postgresql certified professional", "mongodb certified developer", "cassandra certified developer", "kafka certified developer", "elasticsearch certified engineer", "splunk certified admin", "tableau certified associate", "power bi certified data analyst"],
    "degrees": ["bachelor's", "master's", "phd", "mba", "b.a.", "b.s.", "m.s.", "ph.d.", "associate's", "diploma", "certificate"],
    "experience": ["3 years", "4 years", "5 years", "senior", "junior", "entry-level", "manager", "director", "lead", "principal", "executive", "cto", "cio", "ceo"],
    "soft_skills": ["collaboration", "communication", "confidence", "consistency", "cross-cultural sensitivity", "efficiency", "honesty", "initiative", "innovation", "interpersonal relations", "organization", "passion", "prioritization", "team leadership", "problem-solving", "critical thinking", "adaptability", "creativity", "empathy", "decision-making", "time management", "conflict resolution", "leadership", "mentorship", "coaching", "presentation", "public speaking"],
    "employment_details": ["job titles", "employer names", "industry experience", "product experience", "length of employment", "budgets", "staffs", "revenue", "market share", "growth", "achievements", "awards", "recognition"],
    "general_information": ["cities", "states", "ZIP codes", "countries", "honors", "awards", "board positions", "professional affiliations", "volunteer activities", "civic associations", "publications", "patents", "licenses", "certifications", "references", "hobbies", "interests"]
}

def extract_keywords_from_job_description(job_description):
    doc = nlp(job_description)
    entities = [ent.text for ent in doc.ents]
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    yake_keywords = kw_extractor.extract_keywords(job_description)
    yake_keywords = [kw[0] for kw in yake_keywords]
    all_keywords = entities + noun_chunks + yake_keywords
    all_keywords = [kw.lower() for kw in all_keywords if kw.lower() in keyword_dict["skills"] or
                   any(cert.lower() in kw.lower() for cert in keyword_dict["certifications"]) or
                   any(deg.lower() in kw.lower() for deg in keyword_dict["degrees"]) or
                   any(exp.lower() in kw.lower() for exp in keyword_dict["experience"])]
    
    keyword_counts = Counter(all_keywords)
    prioritized_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
    return [kw for kw, count in prioritized_keywords]

job_description = "We are seeking a dedicated and experienced software engineer to join our team. The ideal candidate will have a strong background in software development, with proficiency in programming languages such as Java, Python, or C++. Responsibilities include designing, implementing, and maintaining software solutions, as well as collaborating with cross-functional teams to ensure product delivery. The candidate should possess excellent problem-solving skills and be able to work independently and in a team environment. Experience with agile development methodologies and version control systems is preferred. A bachelor's degree in computer science or a related field is required, and a master's degree is a plus. Join us in our mission to innovate and create cutting-edge software products"

keywords = extract_keywords_from_job_description(job_description)
print("Extracted Keywords:")
for keyword in keywords:
    print(keyword)