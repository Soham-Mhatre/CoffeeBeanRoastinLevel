import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess image
def load_and_preprocess_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, [128, 128])
    image = image / 255.0
    return image

# Function to predict class for a new image
def predict_image(model, image_path):
    image = load_and_preprocess_image(image_path)
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    predictions = model.predict(image)
    predicted_class_index = tf.argmax(predictions[0]).numpy()
    return predicted_class_index, predictions[0]

# Load CSV data for labels
df = pd.read_csv('beandata.csv')

# Display information about the test images
# Example correction based on the DataFrame having only 'filepaths' and 'class_index'

# Create a mapping of class indices to class names
class_index_to_name = {
    0: 'Dark',  
    1: 'Green',
    2: 'Light',
    3: 'Medium'
}

def test_model_on_examples(model, image_paths):
    for image_path in image_paths:
        true_label = df[df['filepaths'] == image_path]['class_index'].values
        if len(true_label) == 0:
            true_label = 'Unknown'
        else:
            true_label = true_label[0]

        predicted_class_index, predictions = predict_image(model, image_path)
        predicted_label = class_index_to_name.get(predicted_class_index, 'Unknown')

        # Print the results
        print(f"Image Path: {image_path}")
        print(f"True Label: {true_label}")
        print(f"Predicted Label: {predicted_label}")
        print(f"Prediction Probabilities: {predictions}")
        print()

        # Display the image
        image = load_and_preprocess_image(image_path).numpy()
        plt.imshow(image)
        plt.title(f"Predicted: {predicted_label}, True: {true_label}")
        plt.axis('off')
        plt.show()


test_image_paths = ['light.png']  

loaded_model = tf.keras.models.load_model('my_image_model.h5')

test_model_on_examples(loaded_model, test_image_paths)
