
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

model = YOLO("yolov8n.pt")

st.title("Object Detection App using YOLOv8")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    img_array = np.array(image)

    
    results = model(img_array)

    
    annotated_image = results[0].plot()

    st.subheader("Detected Objects")
    st.image(annotated_image, use_container_width=True)

    st.subheader("Detection Results")

    for box in results[0].boxes:

        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])

        label = model.names[cls_id]

        st.write(
            f"Object: {label} | Confidence: {confidence:.2f}"
        )
