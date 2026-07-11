import os
import cv2
import numpy as np
import tensorflow as tf
import uuid
import datetime
from werkzeug.utils import secure_filename
from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    session
)

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB limit
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "neuroscan-secret-key-182379")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained Keras model and copy weights to a reconstructed Functional model
# to bypass nested Sequential data augmentation tracing issues during Grad-CAM backpropagation.
original_model = tf.keras.models.load_model("model.keras")

model_input = tf.keras.layers.Input(shape=(128, 128, 3), name='input_layer')
x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', name='conv2d')(model_input)
x = tf.keras.layers.BatchNormalization(name='batch_normalization')(x)
x = tf.keras.layers.MaxPooling2D((2,2), name='max_pooling2d')(x)
x = tf.keras.layers.Dropout(0.25, name='dropout')(x)

x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', name='conv2d_1')(x)
x = tf.keras.layers.BatchNormalization(name='batch_normalization_1')(x)
x = tf.keras.layers.MaxPooling2D((2,2), name='max_pooling2d_1')(x)
x = tf.keras.layers.Dropout(0.25, name='dropout_1')(x)

x = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', name='conv2d_2')(x)
x = tf.keras.layers.BatchNormalization(name='batch_normalization_2')(x)
x = tf.keras.layers.MaxPooling2D((2,2), name='max_pooling2d_2')(x)
x = tf.keras.layers.Dropout(0.30, name='dropout_2')(x)

x = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same', name='conv2d_3')(x)
x = tf.keras.layers.BatchNormalization(name='batch_normalization_3')(x)
x = tf.keras.layers.MaxPooling2D((2,2), name='max_pooling2d_3')(x)
x = tf.keras.layers.Dropout(0.30, name='dropout_3')(x)

x = tf.keras.layers.Flatten(name='flatten')(x)
x = tf.keras.layers.Dense(512, activation='relu', name='dense')(x)
x = tf.keras.layers.Dropout(0.50, name='dropout_4')(x)
x = tf.keras.layers.Dense(256, activation='relu', name='dense_1')(x)
x = tf.keras.layers.Dropout(0.40, name='dropout_5')(x)
x = tf.keras.layers.Dense(128, activation='relu', name='dense_2')(x)
x = tf.keras.layers.Dropout(0.30, name='dropout_6')(x)
model_output = tf.keras.layers.Dense(4, activation='softmax', name='dense_3')(x)

model = tf.keras.models.Model(model_input, model_output)

# Copy weights in-memory
for layer in model.layers:
    try:
        orig_layer = original_model.get_layer(layer.name)
        layer.set_weights(orig_layer.get_weights())
    except ValueError:
        pass

CLASSES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

DESCRIPTIONS = {
    "Glioma": "Glioma is a tumor that develops from glial cells in the brain.",
    "Meningioma": "Meningioma develops in the membranes surrounding the brain.",
    "No Tumor": "No signs of brain tumor detected.",
    "Pituitary": "Pituitary tumors occur in the pituitary gland."
}


def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Construct gradient model
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute gradient of top predicted class
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Calculate gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight Conv activations
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    img = cv2.imread(img_path)
    
    # Resize heatmap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to color map
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    # Save
    cv2.imwrite(cam_path, superimposed_img)


def predict_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("File is not a valid image or could not be decoded.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = CLASSES[predicted_index]
    confidence = round(float(prediction[predicted_index]) * 100, 2)

    probabilities = {
        CLASSES[i]: round(float(prediction[i]) * 100, 2)
        for i in range(4)
    }

    return predicted_class, confidence, probabilities


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    image_name = None
    image_path = None
    gradcam_path = None
    probabilities = None
    description = None
    error_message = None
    accession_no = None
    timestamp = None

    if request.method == "POST":
        file = request.files.get("image")
        
        if not file or file.filename == "":
            error_message = "No file selected for analysis."
        else:
            # Check file size (5MB limit)
            file.seek(0, os.SEEK_END)
            file_length = file.tell()
            file.seek(0)  # Reset pointer

            if file_length == 0:
                error_message = "Uploaded file is empty or corrupted."
            elif file_length > app.config['MAX_CONTENT_LENGTH']:
                error_message = "File size exceeds the 5 MB limit."
            elif not allowed_file(file.filename):
                error_message = "Invalid file type. Supported formats are PNG, JPG, JPEG."
            else:
                secured_name = secure_filename(file.filename)
                _, ext = os.path.splitext(secured_name)
                unique_filename = f"{uuid.uuid4().hex}{ext}"
                path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

                try:
                    file.save(path)
                    prediction, confidence, probabilities = predict_image(path)
                    image_name = secured_name
                    image_path = "uploads/" + unique_filename
                    description = DESCRIPTIONS.get(prediction, "")

                    # Compute Grad-CAM
                    # Preprocess for Grad-CAM
                    img_for_gc = cv2.imread(path)
                    img_for_gc = cv2.cvtColor(img_for_gc, cv2.COLOR_BGR2RGB)
                    img_for_gc = cv2.resize(img_for_gc, (128, 128)) / 255.0
                    img_for_gc = np.expand_dims(img_for_gc, axis=0)

                    # Find last conv layer name
                    last_conv_layer_name = None
                    for layer in reversed(model.layers):
                        if layer.__class__.__name__ == 'Conv2D':
                            last_conv_layer_name = layer.name
                            break

                    if last_conv_layer_name:
                        heatmap = make_gradcam_heatmap(img_for_gc, model, last_conv_layer_name)
                        gc_filename = f"gradcam_{unique_filename}"
                        gc_path_local = os.path.join(app.config['UPLOAD_FOLDER'], gc_filename)
                        save_and_display_gradcam(path, heatmap, gc_path_local)
                        gradcam_path = "uploads/" + gc_filename

                    # Assign report ID and date
                    accession_no = "ACC-" + str(np.random.randint(100000, 999999))
                    timestamp = datetime.datetime.now().strftime("%b %d, %Y at %I:%M %p")

                    # Add prediction history to session
                    history = session.get("prediction_history", [])
                    history.insert(0, {
                        "accession_no": accession_no,
                        "filename": secured_name,
                        "prediction": prediction,
                        "confidence": confidence,
                        "timestamp": timestamp
                    })
                    session["prediction_history"] = history[:5]  # Keep last 5 entries
                    session.modified = True

                except Exception as e:
                    error_message = f"Failed to analyze image: {str(e)}"
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass

    # Retrieve history from session
    prediction_history = session.get("prediction_history", [])

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_name=image_name,
        image_path=image_path,
        gradcam_path=gradcam_path,
        probabilities=probabilities,
        description=description,
        error_message=error_message,
        accession_no=accession_no,
        timestamp=timestamp,
        prediction_history=prediction_history
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    secured_filename = secure_filename(filename)
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        secured_filename
    )


if __name__ == "__main__":
    app.run(debug=True)