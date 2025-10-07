"""Flask API exposing endpoints for upload, job add, matching, and health checks.""" 
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import uuid
from .parser import parse_resume, ParseError
from .db import DatabaseManager
from .matcher import Matcher
from .utils import logger

app = Flask(__name__)
run_with_ngrok(app)

db = DatabaseManager()
matcher = Matcher()

@app.route('/')
def health():
    return jsonify({'status': 'ok', 'message': 'Resume Screening API'})

@app.route('/add_job', methods=['POST'])
def add_job():
    payload = request.get_json()
    if not payload or ('text' not in payload and 'job_description' not in payload):
        return jsonify({'error': 'job payload requires text/job_description'}), 400
    # normalize basic fields
    job = {
        'id': payload.get('id') or str(uuid.uuid4()),
        'title': payload.get('title') or payload.get('job_title'),
        'company': payload.get('company'),
        'text': payload.get('text') or payload.get('job_description')
    }
    jid = db.insert_job(job)
    # refresh matcher
    matcher.fit_jobs(db.list_jobs())
    return jsonify({'job_id': jid}), 201

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    # Accept multipart file upload (recommended) or json path
    if 'file' in request.files:
        file = request.files['file']
        tmp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
        file.save(tmp_path)
        try:
            parsed = parse_resume(tmp_path)
        except ParseError as e:
            return jsonify({'error': str(e)}), 400
    else:
        payload = request.get_json() or {}
        if 'text' not in payload:
            return jsonify({'error': 'Provide file or text in request.'}), 400
        parsed = {'text': payload['text'], 'skills': [], 'education': [], 'experience': []}
    cid = db.insert_candidate({'id': payload.get('id') if 'payload' in locals() else str(uuid.uuid4()),
                               'text': parsed['text'], 'skills': parsed['skills'],
                               'education': parsed['education'], 'experience': parsed['experience']})
    return jsonify({'candidate_id': cid}), 201

@app.route('/match', methods=['POST'])
def match_endpoint():
    payload = request.get_json() or {}
    candidate_id = payload.get('candidate_id')
    top_k = int(payload.get('top_k', 5))
    if not candidate_id:
        return jsonify({'error': 'candidate_id is required'}), 400
    # load candidate
    c = db.conn.execute("SELECT id, text FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
    if not c:
        return jsonify({'error': 'candidate not found'}), 404
    resume_text = c[1]
    # ensure matcher fitted
    if not matcher.job_texts:
        matcher.fit_jobs(db.list_jobs())
    results = matcher.match(resume_text, top_k=top_k)
    # persist matches
    for r in results:
        db.insert_match(candidate_id, r['job_id'], r['score'])
    return jsonify({'matches': results}), 200

if __name__ == '__main__':
    app.run(port=5000)
