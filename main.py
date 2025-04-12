from app import app

# This file is used by gunicorn to import the app variable

if __name__ == '__main__':
    # For local development only
    app.run(host='0.0.0.0', port=5000, debug=True)
