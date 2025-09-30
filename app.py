import os
from typing import List, Tuple

import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import EfficientNetB0
except Exception as e:  # pragma: no cover
    tf = None


#`tamplete` folder for both templates and static assets
app = Flask(__name__, template_folder='tamplete', static_folder='tamplete')


def load_trained_model(model_path: str, num_classes: int):
    if tf is None:
        raise RuntimeError('TensorFlow is not available in the environment.')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found: {model_path}')

    # First try: load full model (architecture + weights)
    try:
        return tf.keras.models.load_model(model_path)
    except Exception:
        # Fallback: build a compatible architecture and load weights by name, skipping mismatches
        input_shape = (224, 224, 3)
        base = EfficientNetB0(include_top=False, weights=None, input_shape=input_shape, pooling='avg')
        x = base.output
        output = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
        model = models.Model(inputs=base.input, outputs=output, name='eye_classifier')
        try:
            model.load_weights(model_path, by_name=True, skip_mismatch=True)
        except Exception as e:
            raise RuntimeError(f'Failed to load weights from {model_path}: {e}')
        return model


def get_target_size(model) -> Tuple[int, int]:
    # Attempt to infer target size from the model input shape (None, H, W, C)
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        _, h, w, _ = shape
        if isinstance(h, int) and isinstance(w, int):
            return int(h), int(w)
    except Exception:
        pass
    # Fallback to a common default
    return 224, 224


def preprocess_image(file_storage, target_size: Tuple[int, int]) -> np.ndarray:
    image = Image.open(file_storage.stream).convert('RGB')
    image = image.resize(target_size)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)
    return array


# Default class names. Update these to match your trained model's classes.
CLASS_NAMES: List[str] = [
    'Cataract',
    'Diabetic Retinopathy',
    'Glaucoma',
    'Normal',
]


MODEL_PATH = os.path.join(os.path.dirname(__file__), 'eye_model_final.h5')
model = load_trained_model(MODEL_PATH, num_classes=len(CLASS_NAMES))
TARGET_SIZE = get_target_size(model)


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided under form field "image".'}), 400

    file_storage = request.files['image']
    if file_storage.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    try:
        input_tensor = preprocess_image(file_storage, TARGET_SIZE)
        preds = model.predict(input_tensor)
        probs = preds[0].tolist()
        if len(probs) != len(CLASS_NAMES):
            # If dimensions mismatch, best-effort to align by truncation/padding
            length = min(len(probs), len(CLASS_NAMES))
            probs = probs[:length]
            classes = CLASS_NAMES[:length]
        else:
            classes = CLASS_NAMES

        best_idx = int(np.argmax(probs))
        response = {
            'prediction': classes[best_idx],
            'probabilities': [
                {'class': cls, 'probability': float(p)} for cls, p in zip(classes, probs)
            ],
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # You can change host/port as needed
    app.run(debug=True)
