from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense

def Meso4():
    x = Input(shape=(256, 256, 3))
    y = Conv2D(8, (3,3), padding='same', activation='relu')(x)
    y = BatchNormalization()(y)
    y = MaxPooling2D(pool_size=(2,2), padding='same')(y)

    y = Conv2D(8, (5,5), padding='same', activation='relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D(pool_size=(2,2), padding='same')(y)

    y = Conv2D(16, (5,5), padding='same', activation='relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D(pool_size=(2,2), padding='same')(y)

    y = Conv2D(16, (5,5), padding='same', activation='relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D(pool_size=(4,4), padding='same')(y)

    y = Flatten()(y)
    y = Dense(16, activation='relu')(y)
    y = Dense(1, activation='sigmoid')(y)
    
    return Model(inputs=x, outputs=y)

def load_model():
    model = Meso4()
    model.load_weights('model_weights.h5')
    return model

def predict(model, image):
    prediction = model.predict(image[np.newaxis, ...])[0][0]
    label = 'Real' if prediction > 0.5 else 'Fake'
    confidence = float(prediction) if label == 'Real' else float(1 - prediction)
    return label, round(confidence, 4)