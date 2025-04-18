
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile

# Load model
model = YOLO("best.pt")  # Make sure best.pt is in same folder or uploaded via UI

st.title("YOLOv11 Object Detection")
st.markdown("Upload an image and get bounding boxes with class labels!")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to NumPy array
    img_np = np.array(image)

    # Perform detection
    results = model(img_np)[0]
    result_img = results.plot()  # Draw bounding boxes

    # Convert back to PIL and display
    result_pil = Image.fromarray(result_img[..., ::-1])
    st.image(result_pil, caption="Detected Image", use_column_width=True)
