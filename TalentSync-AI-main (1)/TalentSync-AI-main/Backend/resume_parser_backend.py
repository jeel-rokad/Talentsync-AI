from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import fitz  # PyMuPDF
import docx
from pathlib import Path
from supabase import create_client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy
from collections import defaultdict, Counter
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime


def load_dotenv(dotenv_path):
    if not os.path.exists(dotenv_path):
        return

    with open(dotenv_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / '.env')

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load spaCy (error handling)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise Exception("Run: python -m spacy download en_core_web_sm")

# ========== API KEYS ==========
API_KEYS = {
    "demo-key-123456789": True,
    "backup-key-987654321": True
}

# Allow overriding valid backend access keys from environment variables
env_keys = os.environ.get('BACKEND_API_KEYS', '')
for key in [k.strip() for k in env_keys.split(',') if k.strip()]:
    API_KEYS[key] = True

BACKEND_API_KEY = os.environ.get('BACKEND_API_KEY')
if BACKEND_API_KEY:
    API_KEYS[BACKEND_API_KEY] = True

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
if not OPENAI_API_KEY:
    print('Warning: OPENAI_API_KEY not set. AI proxy endpoint will be unavailable.')

SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_SERVICE_KEY = os.environ.get('SUPABASE_SERVICE_KEY', '')
SUPABASE_PUBLISHABLE_KEY = os.environ.get('SUPABASE_PUBLISHABLE_KEY', '')
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
else:
    supabase = None
    print('Warning: SUPABASE_URL or SUPABASE_SERVICE_KEY is not set. Supabase persistence disabled.')


def save_resume_to_supabase(parsed_data):
    if not supabase:
        return
    try:
        payload = {
            'name': parsed_data['name'],
            'role': parsed_data['role'],
            'company': parsed_data['company'],
            'skills_by_category': parsed_data['skills_by_category'],
            'skills_text': parsed_data['skills_text'],
            'experience_years': parsed_data['experience_years'],
            'education_percent': parsed_data['education_percent'],
            'is_fresher': parsed_data['is_fresher'],
            'skill_match_percent': parsed_data['skill_match_percent'],
            'experience_score': parsed_data['experience_score'],
            'education_score': parsed_data['education_score'],
            'overall_match_percent': parsed_data['overall_match_percent'],
            'job_description': parsed_data.get('job_description', ''),
            'parsed_at': datetime.now().isoformat()
        }
        result = supabase.table('parsed_resumes').insert(payload).execute()
        if result.error:
            print('Supabase insert error:', result.error)
    except Exception as exc:
        print('Supabase save error:', exc)


def verify_api_key():
    key = request.headers.get('X-API-Key')
    return bool(key and key in API_KEYS)


def openai_chat_completion(prompt, model='gpt-4o-mini', max_tokens=1000):
    if not OPENAI_API_KEY:
        raise RuntimeError('OpenAI API key is not configured in the backend environment.')

    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.2,
        'max_tokens': max_tokens
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    data = response.json()
    if response.status_code != 200:
        raise RuntimeError(data.get('error', {}).get('message', response.text))

    choices = data.get('choices', [])
    if not choices:
        raise RuntimeError('OpenAI returned no completion choices.')

    return choices[0].get('message', {}).get('content', '')

# ========== SKILL DATABASE (200+ skills) ==========
SKILL_CATEGORIES = {
    "Frontend": {
        "skills": ["react", "angular", "vue", "javascript", "html", "css", "typescript", "nextjs", "tailwind", "bootstrap"],
        "subs": {"javascript": ["react", "vue", "angular"]}
    },
    "Backend": {
        "skills": ["nodejs", "python", "django", "flask", "fastapi", "java", "spring", "php", "mongodb", "postgresql"],
        "subs": {"python": ["django", "flask", "fastapi"]}
    },
    "AI/ML": {
        "skills": ["pytorch", "tensorflow", "scikit-learn", "huggingface", "transformers", "llms", "numpy", "pandas"],
        "subs": {"pytorch": ["transformers"]}
    },
    "DevOps": {
        "skills": ["docker", "kubernetes", "aws", "jenkins", "terraform", "ansible", "prometheus"],
        "subs": {}
    }
}

# Skill map
SKILL_MAP = {}
for cat, data in SKILL_CATEGORIES.items():
    for skill in set(data["skills"] + list(data["subs"].keys())):
        SKILL_MAP[skill.lower()] = {"category": cat, "normalized": skill.title()}

# ========== TEXT EXTRACTION ==========
def extract_text(filepath):
    """PDF + DOCX extraction"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.pdf':
        doc = fitz.open(filepath)
        text = " ".join([page.get_text() for page in doc])
        doc.close()
        return text.lower()
    elif ext == '.docx':
        doc = docx.Document(filepath)
        text = " ".join([p.text for p in doc.paragraphs])
        return text.lower()
    return ""

# ========== RESUME PARSING ==========
def parse_resume(text):
    """Full resume parsing"""
    # Name
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    name = lines[0].title() if lines else "Candidate"
    
    # Skills
    doc = nlp(text)
    skill_candidates = []
    for ent in doc.ents:
        if ent.text.lower() in SKILL_MAP:
            skill_candidates.append(ent.text.lower())
    
    # Normalize skills
    skill_count = Counter()
    for skill in skill_candidates:
        if skill in SKILL_MAP:
            info = SKILL_MAP[skill]
            skill_count[(info["normalized"], info["category"])] += 1
    
    categorized = defaultdict(list)
    for (skill_name, cat), count in skill_count.most_common(20):
        if cat in SKILL_CATEGORIES:
            categorized[cat].append({"name": skill_name, "mentions": count})
    
    skills_text = " ".join([name.lower() for name, _ in skill_count])
    
    # Experience
    exp_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?|y\.?)', text, re.I)
    exp_years = sum(float(m[0]) for m in exp_matches)
    
    # Education
    edu_matches = re.findall(r'(?:10th?|12th?|class\s*(?:10|12))\s*[:\-]?\s*(\d+(?:\.\d+)?)', text, re.I)
    edu_scores = [float(m) for m in edu_matches]
    edu_avg = np.mean(edu_scores) if edu_scores else 70.0
    
    # Role & Company
    role_match = re.search(r'(?:software|backend|frontend|ml|ai|devops)\s*(engineer|developer|architect)', text, re.I)
    role = (role_match.group(1).title() + " Engineer") if role_match else "Software Engineer"
    
    company_match = re.search(r'([A-Z][a-zA-Z\s]{1,30})(?:\s*(?:Inc|Corp|LLC|Pvt))?', text)
    company = company_match.group(1).strip() if company_match else "Current Company"
    
    return {
        "name": name,
        "role": role,
        "company": company,
        "skills_by_category": dict(categorized),
        "skills_text": skills_text,
        "experience_years": round(min(exp_years, 30), 1),
        "education_percent": round(edu_avg, 1),
        "is_fresher": exp_years < 1
    }

# ========== API ROUTES ==========
@app.route('/api/parse-resumes', methods=['POST'])
def api_parse_resumes():
    if not verify_api_key():
        return jsonify({"error": "Invalid API key. Use: demo-key-123456789"}), 401
    
    job_desc = request.form.get('job_description', '').lower()
    files = request.files.getlist('resumes')
    
    if not files:
        return jsonify({"error": "No files uploaded"}), 400
    
    resumes = []
    job_skills = []
    doc = nlp(job_desc)
    for ent in doc.ents:
        skill_key = ent.text.lower()
        if skill_key in SKILL_MAP:
            job_skills.append(SKILL_MAP[skill_key]["normalized"])
    
    job_skills_text = " ".join(job_skills).lower()
    
    for file in files:
        if file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                text = extract_text(filepath)
                if not text.strip():
                    continue
                
                resume = parse_resume(text)
                
                # Skill matching
                if job_skills_text:
                    vectorizer = TfidfVectorizer(stop_words='english')
                    texts = [resume['skills_text'], job_skills_text]
                    tfidf_matrix = vectorizer.fit_transform(texts)
                    skill_match = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                else:
                    skill_match = 0
                
                # Scores
                skill_score = min(skill_match * 100, 100)
                exp_score = min(resume['experience_years'] * 8, 90)
                edu_score = resume['education_percent'] * 0.8
                overall = skill_score * 0.5 + exp_score * 0.3 + edu_score * 0.2
                
                resume.update({
                    "skill_match_percent": round(skill_score, 1),
                    "experience_score": round(exp_score, 1),
                    "education_score": round(edu_score, 1),
                    "overall_match_percent": round(overall, 1)
                })
                
                save_resume_to_supabase({**resume, 'job_description': job_desc})
                resumes.append(resume)
                
            except Exception as e:
                print(f"Error: {e}")
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
    
    resumes.sort(key=lambda x: x["overall_match_percent"], reverse=True)
    
    return jsonify({
        "success": True,
        "resumes": resumes,
        "job_required_skills": job_skills[:12],
        "job_description": job_desc,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/skill-graph/<category>')
def api_skill_graph(category):
    if not verify_api_key():
        return jsonify({"error": "Invalid API key"}), 401
    
    category = category.title().replace("Ai", "AI/ML").replace("Ai/Ml", "AI/ML")
    if category not in SKILL_CATEGORIES:
        return jsonify({"error": f"Category '{category}' not found"}), 400
    
    data = SKILL_CATEGORIES[category]
    nodes = [{"id": category, "group": 1, "size": 25, "isCentral": True}]
    links = []
    
    # Main skills
    for i, skill in enumerate(data["skills"][:8]):
        nodes.append({"id": skill.title(), "group": 2, "size": 15})
        links.append({"source": category, "target": skill.title(), "value": 2})
    
    # Sub-skills
    for main_skill, subs in list(data.get("subs", {}).items())[:3]:
        nodes.append({"id": main_skill.title(), "group": 3, "size": 12})
        links.append({"source": category, "target": main_skill.title(), "value": 1.5})
        for sub in subs[:4]:
            nodes.append({"id": sub.title(), "group": 4, "size": 10})
            links.append({"source": main_skill.title(), "target": sub.title(), "value": 1})
    
    return jsonify({"nodes": nodes, "links": links})

@app.route('/api/categories')
def api_categories():
    if not verify_api_key():
        return jsonify({"error": "Invalid API key"}), 401
    return jsonify(list(SKILL_CATEGORIES.keys()))

@app.route('/api/config')
def api_config():
    if not verify_api_key():
        return jsonify({"error": "Invalid API key"}), 401

    return jsonify({
        "supabase_url": SUPABASE_URL,
        "supabase_publishable_key": SUPABASE_PUBLISHABLE_KEY,
        "openai_enabled": bool(OPENAI_API_KEY)
    })

@app.route('/api/ai', methods=['POST'])
def api_ai():
    if not verify_api_key():
        return jsonify({"error": "Invalid API key"}), 401

    payload = request.get_json(silent=True) or {}
    prompt = payload.get('prompt', '').strip()
    model = payload.get('model', 'gpt-4o-mini')

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    try:
        response_text = openai_chat_completion(prompt, model=model)
        return jsonify({"success": True, "response": response_text})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500

@app.route('/health', methods=['GET', 'POST'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "version": "1.0.0",
        "api_key_count": len(API_KEYS),
        "openai_enabled": bool(OPENAI_API_KEY),
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 HACKATHON RESUME PARSER BACKEND")
    print("📱 Port: http://localhost:5000")
    print("🔑 API Key: demo-key-123456789")
    print("📋 Endpoints:")
    print("   POST /api/parse-resumes")
    print("   GET  /api/skill-graph/<category>")
    print("   GET  /api/categories")
    print("   POST /api/ai")
    print("   GET/POST /health")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)