import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load model
        model = load_model(os.path.join("model", "model.h5"))

        # Load and preprocess the image
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0  # Normalize to [0, 1]
        test_image = np.expand_dims(test_image, axis=0)

        # Make prediction
        raw_predictions = model.predict(test_image)
        result = np.argmax(raw_predictions, axis=1)

        # Debugging outputs
        print(f"Raw Predictions: {raw_predictions}")
        print(f"Predicted Class: {result[0]}")

        # Interpret the result
        if result[0] == 1:
            prediction = 'Normal'
        else:
            prediction = 'Adenocarcinoma Cancer'

        return [{"image": prediction}]
