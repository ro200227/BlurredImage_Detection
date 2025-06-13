#importing libraries
import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

# Load model
model = joblib.load("blur_detection_model.pkl")

# Feature extraction
def compute_features(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    image = cv2.resize(image, (256, 256))

    laplacian = cv2.Laplacian(image, cv2.CV_64F).var()

    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    magnitude = 20 * np.log(np.abs(fft_shift) + 1e-10)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols))
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
    fft_mean = np.mean(magnitude * mask)

    edges = cv2.Canny(image, 50, 100)
    edge_density = np.mean(edges > 0)

    return [laplacian, fft_mean, edge_density]

# Streamlit UI
st.set_page_config(page_title="Blurred Image Detection", layout="centered")
st.title("Blurred Image Detection App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

    features = compute_features(image)
    prediction = model.predict([features])[0]

    classes = {0: "Sharp", 1: "Defocused", 2: "Motion Blur"}
    st.markdown(f"### 📌 Prediction: `{classes[prediction]}`")
