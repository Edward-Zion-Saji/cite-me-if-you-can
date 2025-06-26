from flask import Flask, render_template, request, jsonify, redirect, url_for
import requests
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['BACKEND_URL'] = os.environ.get('BACKEND_URL', 'http://localhost:8000')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return redirect(url_for('chat'))

@app.route('/chat')
def chat():
    return render_template('index.html', active_page='chat')

@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.json
    try:
        response = requests.post(
            f"{app.config['BACKEND_URL']}/api/chat",
            json={"query": data.get('query')}
        )
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Send to backend
        with open(filepath, 'rb') as f:
            files = {'file': (filename, f, 'application/json')}
            response = requests.put(
                f"{app.config['BACKEND_URL']}/api/upload",
                files=files
            )
        
        # Clean up
        os.remove(filepath)
        return jsonify(response.json())

@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    try:
        print(f"Searching for: {data.get('query')}")
        response = requests.post(
            f"{app.config['BACKEND_URL']}/api/similarity_search",
            json={"query": data.get('query')}
        )
        response.raise_for_status()
        results = response.json()
        print(f"Search results: {len(results)} items")
        if results:
            print(f"First result: {results[0].keys()}")
        return jsonify(results)
    except requests.exceptions.RequestException as e:
        print(f"Search error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/citations')
def get_citations():
    try:
        response = requests.get(f"{app.config['BACKEND_URL']}/api/citations/stats")
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
