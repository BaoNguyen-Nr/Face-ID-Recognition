import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import uuid
from recognize_face import FaceRecognizer

# Đường dẫn FaceBank
FACEBANK_PATH = 'facebank'
MIN_IMAGES_REQUIRED = 3

# Khởi tạo recognizer
recognizer = FaceRecognizer()

st.set_page_config(page_title="Face Recognition", layout="centered")
st.title("📷 Nhận Diện Khuôn Mặt Qua Webcam")

tab1, tab2 = st.tabs(["🔍 Nhận Diện", "➕ Thêm Người Mới"])

with tab1:
    camera_image = st.camera_input("📸 Chụp ảnh để nhận diện:")

    if camera_image is not None:
        image_pil = Image.open(camera_image)
        image_rgb = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Nhận diện khuôn mặt
        bboxes, names, annotated_img = recognizer.recognize_faces(image_bgr.copy(), draw=True)

        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, channels="RGB", caption="Ảnh đã nhận diện")

        if names:
            st.subheader("🧑‍💻 Kết quả nhận diện:")
            for i, name in enumerate(names):
                st.write(f"Face {i+1}: **{name}**")
        else:
            st.warning("❌ Không nhận diện được khuôn mặt nào.")

with tab2:
    st.subheader("👤 Thêm người mới vào hệ thống")
    person_name = st.text_input("Nhập tên người mới:")
    add_image = st.camera_input("📸 Chụp ảnh để thêm")

    if 'images_collected' not in st.session_state:
        st.session_state.images_collected = []

    if person_name and add_image is not None:
        image_pil = Image.open(add_image)
        image_rgb = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Lưu ảnh tạm vào bộ nhớ session
        st.session_state.images_collected.append(image_bgr)
        st.success(f"✅ Đã thêm ảnh ({len(st.session_state.images_collected)}/3)")

    if len(st.session_state.images_collected) >= MIN_IMAGES_REQUIRED:
        if st.button("💾 Lưu vào FaceBank"):
            save_dir = os.path.join(FACEBANK_PATH, person_name)
            os.makedirs(save_dir, exist_ok=True)

            for idx, img in enumerate(st.session_state.images_collected):
                filename = f"{uuid.uuid4().hex[:8]}.jpg"
                cv2.imwrite(os.path.join(save_dir, filename), img)

            st.success(f"🎉 Đã lưu {len(st.session_state.images_collected)} ảnh cho **{person_name}**.")
            st.session_state.images_collected = []

            # Cập nhật facebank
            recognizer.update_facebank()
