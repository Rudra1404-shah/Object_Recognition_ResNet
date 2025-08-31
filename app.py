# app.py
import gradio as gr
import numpy as np
import tensorflow as tf

# 1) Load model once
MODEL_PATH = "cifar10_resnet50.keras"  # ensure this file is in the same directory
model = tf.keras.models.load_model(MODEL_PATH)

# 2) CIFAR-10 class names (ordered to match training labels/order)
CLASS_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# 3) Preprocess function
# If you trained with ResNet50 preprocessing and 224x224 inputs, keep this:
TARGET_SIZE = (224, 224)
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess  # ResNet50/ResNet
# For ResNet50V2 use: from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess

def preprocess_image(img: np.ndarray) -> np.ndarray:
    # img arrives as HxWxC RGB uint8 from Gradio
    img = tf.image.resize(img, TARGET_SIZE).numpy()
    img = np.expand_dims(img, axis=0)  # to NCHW-like batch: (1,H,W,C)
    img = resnet_preprocess(img)       # match training preprocessing
    return img

# 4) Prediction function returning a dict label->prob for Gradio Label output
def predict(image: np.ndarray):
    if image is None:
        return {}
    x = preprocess_image(image)
    preds = model.predict(x, verbose=0)  # shape (10,)
    # Softmax usually already applied by final layer; if not, apply:
    if preds.ndim == 1 and (preds.min() < 0 or preds.max() > 1):
        preds = tf.nn.softmax(preds).numpy()
    probs = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    return probs

# 5) Build Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),   # returns RGB numpy array
    outputs=gr.Label(num_top_classes=3),
    title="CIFAR-10 Classifier",
    description="Upload an image to classify into CIFAR-10 classes."
)

if __name__ == "__main__":
    demo.launch()  # add share=True to create a public link
