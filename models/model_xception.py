import numpy as np
import tensorflow as tf
from utils.preprocessing import preprocess

model = tf.keras.models.load_model('saved_models/xception.keras')
class_names = [
    "Actinic Keratosis (AK)",
    "Basal Cell Carcinoma (BCC)",
    "Benign Keratosis Lesions  (BKL)",
    "Dermatofibroma (DF)",
    "Melanoma (MEL)",
    "Melanocytic Nevus (NV)",
    "Vascular Lesions (VASC)"
]

def preprocess_image(image):
    preprocessed = preprocess(image)
    return preprocessed

def predict_skin_disease(image):
    try:
        image_np = np.array(image.convert('RGB'))
        img_array = preprocess_image(image_np)
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)[0]
        top_predictions = decode_predictions(predictions)
        return top_predictions
    except Exception as e:
        print(f"Error in predict_skin_disease: {e}")
        return []

def decode_predictions(predictions):
    top_indices = predictions.argsort()[-2:][::-1]
    top_preds = [
        {"class": class_names[i], "probability": float(predictions[i])}
        for i in top_indices
    ]
    return top_preds