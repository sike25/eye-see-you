import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import argparse
import os
from tensorflow import keras

feature_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

class MultiLabelAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="multilabel_accuracy", **kwargs):
        super(MultiLabelAccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)
        y_true_bool = tf.cast(y_true, tf.bool)
        y_pred_bool = tf.cast(y_pred, tf.bool)
        intersection = tf.reduce_sum(tf.cast(tf.logical_and(y_true_bool, y_pred_bool), tf.float32), axis=1)
        union = tf.reduce_sum(tf.cast(tf.logical_or(y_true_bool, y_pred_bool), tf.float32), axis=1)
        accuracy = intersection / union
        self.total.assign_add(tf.reduce_sum(accuracy))
        self.count.assign_add(tf.cast(tf.size(accuracy), tf.float32))

    def result(self):
        return self.total / self.count

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

# Function to load the model
def load_model(path):
    if not os.path.exists(path):
        raise ValueError("Specified model path does not exist.")
    try:
        model = tf.keras.models.load_model(path, custom_objects={'MultiLabelAccuracy': MultiLabelAccuracy})
        return model
    except Exception as e:
        raise ValueError("Failed to load the model. Please ensure the path is correct and the file is a valid Keras model.") from e
    
def apply_thresholds(y_probs, thresholds = 0.55):
    # Convert the prediction probabilites to 0 or 1 using the threshold established in the model
    return (y_probs >= thresholds).astype(int)

# Function to classify the image using the loaded model
def classify_image(image):

    print("##### Image shape -------->", image.shape)
    
    image = np.resize(image, (150, 150, 3))

    # Normalize the image (adjust this preprocessing as necessary)
    scaler = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)

    image_array = scaler(image) # Example normalization
    
    # Model predictions
    probabilities = model.predict(np.array([image_array]))
    prediction = apply_thresholds(y_probs=probabilities)
    
    return ", ".join([feature_names[i] for i in range(prediction.shape[1]) if prediction[0, i] == 1])

# Parse command line arguments
parser = argparse.ArgumentParser(description='Image Classification Model Path')
parser.add_argument('model_path', type=str, help='Path to the pre-trained model file')
args = parser.parse_args()
# Check if the model path is provided and is valid
if not args.model_path:
    parser.error("No model path provided.")
model = load_model(args.model_path)

gr.Image()

# Create the Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(),
    outputs=gr.Label(),
    title="Image Classification",
    description="Upload an image and the model will classify it."
)

# Launch the application
iface.launch()
