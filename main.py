import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile

# Load model
model = YOLO("best.pt")  # Ensure this file is present or uploaded

st.title("YOLOv11 Object Detection - CSE_C19")


uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to NumPy (OpenCV format)
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Run prediction
    results = model(img_cv, conf=0.5)

    # Draw bounding boxes manually with custom font size
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # Draw rectangle
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Prepare label text
        text = f"{label} {conf:.2f}"
        font_scale = 0.3
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Label position
        text_x = x1
        text_y = max(y1 - 10, th + 5)

        # Background for text
        cv2.rectangle(img_cv, (text_x, text_y - th), (text_x + tw, text_y), (0, 255, 0), -1)
        cv2.putText(img_cv, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # Convert BGR to RGB
    img_result_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(img_result_rgb)

    st.image(result_pil, caption="Detected Image", use_column_width=True)
