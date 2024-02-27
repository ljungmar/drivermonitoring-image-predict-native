import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

# Load pre-trained model from .h5 file
model = load_model('./saved_models/trained_model_64x64.h5')  # Replace 'your_model.h5' with the path to your trained model file

def main(image_path):
    # Load and preprocess the input image
    img = image.load_img(image_path, target_size=(64, 64))  # Adjust target_size as needed
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction using the pre-trained model
    prediction = model.predict(img_array)

    # Interpret the prediction and provide the result to the user
    if prediction[0][0] < 0.5:
        print('distracted')
    else:
        print('focused')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Driver Monitoring Prediction')
    parser.add_argument('-p', '--image_path', required=True, help='Path to the uploaded image')
    args = parser.parse_args()
    print(args)
    print('trained models will be found in saved_models/')
    tf.config.run_functions_eagerly(True)
    main(args.image_path)