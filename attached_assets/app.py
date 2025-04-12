from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os, datetime, numpy as np
from mesonet import load_model, predict
from preprocess import preprocess_image
from database import log_result


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    img = preprocess_image(filepath)
    result = predict(model, img)
    log_result(filename, result)
    return render_template('result.html', result=result, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)

