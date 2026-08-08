from flask import Flask, send_from_directory, jsonify
import os

app = Flask(__name__)

# Directory where our encoded video files are stored
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIDEO_DIR = os.path.join(BASE_DIR, "data", "encoded_video")

@app.route('/')
def index():
    """Simple homepage"""
    return """
    <html>
    <body>
        <h1>DASH Streaming Server</h1>
        <p>Available endpoints:</p>
        <ul>
            <li><a href="/manifest.mpd">/manifest.mpd</a> - DASH manifest</li>
            <li><a href="/status">/status</a> - Server status & files</li>
        </ul>
    </body>
    </html>
    """

@app.route('/manifest.mpd')
def serve_manifest():
    """Serve the DASH manifest file"""
    response = send_from_directory(VIDEO_DIR, 'manifest.mpd', mimetype='application/dash+xml')
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/video/<path:filename>')
def serve_video(filename):
    """Serve video segments"""
    response = send_from_directory(VIDEO_DIR, filename)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/status')
def server_status():
    """Check what files are available"""
    files = []
    for root, dirs, filenames in os.walk(VIDEO_DIR):
        for f in filenames:
            files.append(os.path.join(root, f))
    return jsonify({
        'status': 'running',
        'files_available': len(files),
        'files': files
    })

if __name__ == '__main__':
    print(f"Starting streaming server...")
    print(f"Serving files from: {os.path.abspath(VIDEO_DIR)}")
    print(f"Access at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
