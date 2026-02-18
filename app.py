import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# Set tajuk kat browser
st.set_page_config(page_title="RSP AI Detector", layout="wide")
st.title("✌️ Rock Paper Scissors AI")
st.write("Tunjuk tangan kau kat kamera, biar AI teka!")

# 1. Load model (Ganti path kalau lain)
@st.cache_resource
def load_model():
    return YOLO('runs/detect/train/weights/best.pt')

model = load_model()

# 2. Setup Kamera
img_file_buffer = st.camera_input("Ambil gambar tangan kau")

if img_file_buffer is not None:
    # Tukar gambar jadi format yang OpenCV faham
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # 3. Predict guna YOLO
    results = model(cv2_img, conf=0.5)

    # 4. Tunjuk hasil
    annotated_frame = results[0].plot()
    # YOLO pakai BGR, Streamlit pakai RGB
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    st.image(annotated_frame_rgb, caption="Hasil Tekaan AI")

    # Ambil nama class yang dia jumpa
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            st.success(f"AI nampak: **{label.upper()}**!")