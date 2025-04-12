from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import logging
import os

logger = logging.getLogger(__name__)

# Load the face cascade classifier
face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
if os.path.exists(face_cascade_path):
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
else:
    logger.warning(f"Face cascade file not found at {face_cascade_path}")
    face_cascade = None

def preprocess_image(path):
    """
    Preprocess an image for MesoNet model.
    
    Args:
        path: Path to the image file
        
    Returns:
        Preprocessed image as a numpy array of shape (256, 256, 3)
    """
    try:
        # Load and resize the image
        img = image.load_img(path, target_size=(256, 256))
        img = image.img_to_array(img)
        
        # Normalize pixel values to [0, 1]
        img = img / 255.0
        
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image {path}: {e}")
        raise

def is_valid_image(path):
    """
    Check if an image contains a detectable face.
    
    Args:
        path: Path to the image file
        
    Returns:
        Boolean indicating if a face was detected
    """
    if face_cascade is None:
        # If face detection is unavailable, assume image is valid
        return True
    
    try:
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
