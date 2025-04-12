from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense
import os
import logging

logger = logging.getLogger(__name__)

def Meso4():
    """
    Implementation of the MesoNet model architecture for deepfake detection.
    
    Returns:
        A Keras Model with the MesoNet architecture
    """
    x = Input(shape=(256, 256, 3))
    
    # First convolutional block
    y = Conv2D(8, (3,3), padding='same', activation='relu')(x)
    y = BatchNormalization()(y)
    y = MaxPooling2D(pool_size=(2,2), padding='same')(y)
    
    # Second convolutional block
    y = Conv2D(8, (5,5), padding='same', activation='relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D(pool_size=(2,2), padding='same')(y)
    
    # Third convolutional block
    y = Conv2D(16, (5,5), padding='same', activation='relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D(pool_size=(2,2), padding='same')(y)
    
    # Fourth convolutional block
    y = Conv2D(16, (5,5), padding='same', activation='relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D(pool_size=(4,4), padding='same')(y)
    
    # Fully connected layers
    y = Flatten()(y)
    y = Dense(16, activation='relu')(y)
    y = Dense(1, activation='sigmoid')(y)
    
    return Model(inputs=x, outputs=y)

def load_model():
    """
    Loads the MesoNet model with pre-trained weights.
    
    Returns:
        A compiled Keras model ready for inference
    """
    model = Meso4()
    
    # Attempt to load weights, try multiple locations
    weight_paths = ['model_weights.h5', 'static/model/model_weights.h5']
    
    for path in weight_paths:
        if os.path.exists(path):
            try:
                model.load_weights(path)
                logger.info(f"Model weights loaded from {path}")
                return model
            except Exception as e:
                logger.error(f"Error loading weights from {path}: {e}")
    
    # If model weights are not found, warn but return the model anyway
    # In a real app, you might want to handle this differently
    logger.warning("No model weights found. Using uninitialized model.")
    return model

def predict(model, image):
    """
    Makes a prediction using the MesoNet model on a preprocessed image.
    
    Args:
        model: The MesoNet model
        image: A preprocessed image of shape (256, 256, 3)
        
    Returns:
        A tuple of (prediction_label, confidence_score)
    """
    # Get the prediction
    result = model.predict(image[np.newaxis, ...])
    confidence = float(result[0][0])
    
    # Return result and confidence
    if confidence > 0.5:
        return 'Real', confidence
    else:
        return 'Fake', 1.0 - confidence
