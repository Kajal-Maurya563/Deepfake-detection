import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import datetime
import random
import cv2
from database import log_result, get_recent_results, init_db

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "deepfake_detection_secret")

# Configure database connection (using PostgreSQL)
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the database
init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_valid_image(path):
    """
    Check if an image contains a detectable face.
    
    Args:
        path: Path to the image file
        
    Returns:
        Boolean indicating if a face was detected
    """
    try:
        # Load the face cascade classifier
        face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Load the image in grayscale for face detection
        img = cv2.imread(path)
        if img is None:
            logger.error(f"Could not read image file: {path}")
            return False
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Return True if at least one face is detected
        return len(faces) > 0
    except Exception as e:
        logger.error(f"Error checking image validity {path}: {e}")
        # In case of error, assume image is valid to allow processing
        return True

@app.route('/')
def home():
    # Get recent results for the dashboard
    recent_results = get_recent_results(5)
    return render_template('index.html', recent_results=recent_results)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('home'))
    
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload a JPG, JPEG, or PNG file.', 'danger')
        return redirect(url_for('home'))
    
    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')
        unique_filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Check if image is valid (contains a face)
        if not is_valid_image(filepath):
            flash('No face detected in the image. Please upload an image with a clearly visible face.', 'warning')
            return redirect(url_for('home'))
        
        # For demo purposes, use a simple random prediction
        # In a real app, this would use the actual MesoNet model
        result = random.choice(['Real', 'Fake'])
        confidence = random.uniform(0.65, 0.95)
        
        # Log the result to database
        log_result(unique_filename, result, confidence)
        
        return render_template('result.html', 
                              result=result, 
                              confidence=confidence,
                              filename=unique_filename)
    
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        flash(f'An error occurred: {str(e)}', 'danger')
        return redirect(url_for('home'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how_it_works')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/convolution')
def convolution():
    return render_template('convolution.html')

@app.route('/gallery')
def gallery():
    return render_template('gallery.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 16MB.', 'danger')
    return redirect(url_for('home'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
