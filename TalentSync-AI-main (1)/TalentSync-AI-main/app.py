import os
import sys
from flask import send_from_directory

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT, 'Backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from resume_parser_backend import app

@app.route('/')
def index():
    return send_from_directory(ROOT, 'talentsync-ai.html')

@app.route('/Frontend/<path:path>')
def frontend_static(path):
    return send_from_directory(os.path.join(ROOT, 'Frontend'), path)

@app.route('/<path:path>')
def root_static(path):
    if os.path.exists(os.path.join(ROOT, path)):
        return send_from_directory(ROOT, path)
    return send_from_directory(ROOT, 'talentsync-ai.html')

if __name__ == '__main__':
    print('Starting TalentSync AI root server...')
    app.run(debug=True, host='0.0.0.0', port=5000)
