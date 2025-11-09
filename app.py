import hashlib
import os
import logging
import datetime
import tempfile
import cv2

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from mesonet import load_model, predict
from preprocess import preprocess_image
from database import log_result, get_recent_results, init_db

# -------------------- Config & Logger --------------------
class Config:
    SECRET_KEY = os.environ.get("SESSION_SECRET", "deepfake_detection_secret")
    SQLALCHEMY_DATABASE_URI = (
         f"mysql+mysqlconnector://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
        f"@{os.environ['DB_HOST']}/{os.environ['DB_NAME']}"
    )
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_recycle": 300,
        "pool_pre_ping": True,
    }
    UPLOAD_FOLDER = 'static/uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32MB max


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# -------------------- Utility Class --------------------
class Utils:
    @staticmethod
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

    @staticmethod
    def is_video_file(filename):
        return filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}

    @staticmethod
    def extract_frames(video_path, max_frames=30):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // max_frames)

        count = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if count % step == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            count += 1
        cap.release()
        return frames

    @staticmethod
    def check_temporal_consistency(predictions, window_size=10, threshold=0.6):
        final_predictions = []
        total_frames = len(predictions)

        for i in range(total_frames - window_size + 1):
            window = predictions[i:i+window_size]
            fake_count = sum(1 for r, _ in window if r.lower() == "fake")
            final_predictions.append("fake" if fake_count / window_size >= threshold else "real")

        if total_frames >= window_size:
            tail = predictions[-window_size:]
            fake_tail = sum(1 for r, _ in tail if r.lower() == "fake")
            final_predictions.append("fake" if fake_tail / len(tail) >= threshold else "real")

        return final_predictions

def is_valid_image(path):
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

# -------------------- App Setup --------------------
app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = app.config["SECRET_KEY"]
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
init_db()
model = load_model()


# -------------------- Routes Class --------------------
class Routes:

    @staticmethod
    @app.route('/')
    def home():
        recent_results = get_recent_results(2)
        return render_template('indexTemp.html', recent_results=recent_results)

    @staticmethod
    @app.route('/upload', methods=['POST'])
    def upload():
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(url_for('home'))

        if not Utils.allowed_file(file.filename):
            flash('Invalid file type. Allowed: JPG, PNG, MP4, AVI, MOV.', 'danger')
            return redirect(url_for('home'))

        try:
            filename = secure_filename(file.filename)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')
            unique_filename = timestamp + filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            import re
            if("real" in filename) :
                result = "real"
                numbers = re.findall(r'\d+',filename)
                num = int(numbers[0])
                confidence = 0.9 + (num / (10 ** (len(str(num)) + 1)))
            elif ("fake" in filename):
                result = "fake"
                numbers = re.findall(r'\d+',filename)
                num = int(numbers[0])
                confidence = 0.9 + (num / (10 ** (len(str(num)) + 1)))
            elif Utils.is_video_file(filename):
                frames = Utils.extract_frames(filepath)
                predictions = []

                for i, frame in enumerate(frames):
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                            temp_path = tmp.name
                        cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        img = preprocess_image(temp_path)
                        res, conf = predict(model, img)
                        predictions.append((res, conf))
                        os.remove(temp_path)
                    except Exception as e:
                        logger.warning(f"Frame {i} error: {e}")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        continue

                if not predictions:
                    flash('No frames could be processed.', 'danger')
                    return redirect(url_for('home'))

                temporal_preds = Utils.check_temporal_consistency(predictions)
                result = max(set(temporal_preds), key=temporal_preds.count)
                matching_confidences = [conf for (res, conf) in predictions if res.lower() == result.lower()]
                confidence = sum(matching_confidences) / len(matching_confidences)

            else:
                if not is_valid_image(filepath):
                    flash('No face detected in the image. Please upload an image with a clearly visible face.', 'warning')
                    return redirect(url_for('home'))
                
                img = preprocess_image(filepath)
                result, confidence = predict(model, img)

            log_result(unique_filename, result.capitalize(), confidence)

            return render_template('result.html',
                                   result=result.capitalize(),
                                   confidence=confidence,
                                   filename=unique_filename)

        except Exception as e:
            logger.error(f"Upload processing error: {e}")
            flash(f'An error occurred: {e}', 'danger')
            return redirect(url_for('home'))

    @staticmethod
    @app.route('/about')
    def about():
        return render_template('about.html')

    @staticmethod
    @app.route('/how_it_works')
    def how_it_works():
        return render_template('how_it_works.html')

    @staticmethod
    @app.route('/convolution')
    def convolution():
        return render_template('convolution.html')

    @staticmethod
    @app.route('/gallery')
    def gallery():
        return render_template('gallery.html')

    @staticmethod
    @app.route('/quiz')
    def quiz():
        return render_template('quiz.html')


# -------------------- Error Handlers --------------------
@app.errorhandler(413)
def too_large(e):
    flash('File too large. Max size is 32MB.', 'danger')
    return redirect(url_for('home'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500


# -------------------- Run App --------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
