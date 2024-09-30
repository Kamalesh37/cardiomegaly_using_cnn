import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array 
# 6. Classification function
def classify_image(model_path, image_path):
    """
    This function loads a saved model and performs classification on a single image.
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to match training data

    # Perform prediction
    prediction = model.predict(img_array)

    # Class interpretation
    if prediction[0] > 0.5:
        return "Cardiomegaly disease"
    else:
        return "No Cardiomegaly disease"

# 7. Usage example for classification
model_path = 'C:/model/my_model.h5'
image_path = "C:/Users/Admin/OneDrive/Desktop/64.png"

# Predict the class of the image
result = classify_image(model_path, image_path)
print(f"Classification Result: {result}")