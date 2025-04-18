import cv2
from PIL import Image
import tensorflow as tf

def preprocess_image1(image):
    try:
        grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
        _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        final_image = cv2.inpaint(image, threshold, 1, cv2.INPAINT_TELEA)
        gaussian = cv2.GaussianBlur(final_image, (0, 0), 2.0)
        processed=cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        return processed_rgb
    except Exception as e:
        print(f"Error in hair removal: {str(e)}")
        return image

def preprocess_image2(image):
    img = Image.fromarray(image).resize((299, 299)) 
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    return img_array

def preprocess(image):
    image = preprocess_image1(image)
    image = preprocess_image2(image)
    return image