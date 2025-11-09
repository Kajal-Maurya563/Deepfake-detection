from flask import render_template, request, redirect, url_for, flash
from deepfake_model import DeepfakeModel
from utils.file_utils import FileUtils
from utils.video_utils import VideoUtils
from database import log_result, get_recent_results

model = DeepfakeModel()

def create_routes(app):
    @app.route('/')
    def home():
        recent_results = get_recent_results(2)
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

        if not FileUtils.allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
            flash('Invalid file type. Allowed: JPG, PNG, MP4, AVI, MOV.', 'danger')
            return redirect(url_for('home'))

        try:
            filepath = FileUtils.save_file(file, app.config['UPLOAD_FOLDER'])

            if VideoUtils.is_video_file(file.filename):
                frames = VideoUtils.extract_frames(filepath)
                predictions = []

                for frame in frames:
                    temp_path = VideoUtils.preprocess_video_frame(frame)
                    result, confidence = model.predict(temp_path)
                    predictions.append((result, confidence))
                    os.remove(temp_path)

                if not predictions:
                    flash('No frames could be processed.', 'danger')
                    return redirect(url_for('home'))

                temporal_preds = check_temporal_consistency(predictions)
                result = max(set(temporal_preds), key=temporal_preds.count)
                confidence = sum(conf for _, conf in predictions) / len(predictions)

            else:
                result, confidence = model.predict(filepath)

            log_result(filepath, result, confidence)

            return render_template('result.html', result=result.capitalize(), confidence=round(confidence, 2), filename=filepath)

        except Exception as e:
            logger.error(f"Upload processing error: {e}")
            flash(f'An error occurred: {e}', 'danger')
            return redirect(url_for('home'))
