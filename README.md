# Face-ID-Recognition
A face recognition system using YOLOv8 and ResNet with triplet loss and clustering-based facebank
# Face ID Recognition System

Hệ thống nhận diện khuôn mặt sử dụng Deep Learning, hỗ trợ:
- Phát hiện và nhúng khuôn mặt bằng YOLOv8 + ResNet.
- Nhận diện khuôn mặt với cơ sở dữ liệu `facebank`.
- Giao diện Streamlit để sử dụng webcam, upload ảnh, và thêm người mới.
- Hệ thống được xây dựng lại toàn bộ pipeline (Dataset, Model, Loss, Training loop).

## 🚀 Demo
<img src="demo/demo.gif" width="600"/>

## 📁 Cấu trúc thư mục

FACE ID/
│
├── facebank.pkl # Cơ sở dữ liệu embedding
├── best_model.pt # Mô hình nhúng khuôn mặt đã huấn luyện
├── face_detector.py # Phát hiện khuôn mặt (YOLOv8)
├── face_embedder.py # Mô hình nhúng (ResNet50)
├── register_facebank.py # Tạo hoặc cập nhật facebank
├── inference.py # Nhận diện từ ảnh đầu vào hoặc webcam
├── app_streamlit.py # Giao diện Streamlit
├── requirements.txt # Thư viện cần thiết
├── README.md
└── ...

## ⚙️ Cài đặt

```bash
# Clone repo
git clone https://github.com/BaoNguyen-Nr/Face-ID-Recognition.git
cd Face-ID-Recognition

# Tạo môi trường và cài đặt requirements
python -m venv venv
venv\Scripts\activate   # Trên Windows
pip install -r requirements.txt
Cách sử dụng
1. Đăng ký dữ liệu vào facebank
Chạy script để tạo facebank từ ảnh trong thư mục data/train (mỗi người một thư mục riêng):

bash
Copy
Edit
python register_facebank.py
2. Giao diện sử dụng
Chạy app Streamlit:

bash
Copy
Edit
streamlit run app_streamlit.py
Tại đây bạn có thể:

Quét khuôn mặt qua webcam.

Upload ảnh để dự đoán.

Gán nhãn và thêm người mới vào facebank (cần ít nhất 3 ảnh).

💡 Công nghệ sử dụng
Python

PyTorch

YOLOv8 (Ultralytics)

ResNet (nhúng)

Streamlit

OpenCV

📌 TODO
 Nhận diện qua webcam

 Upload ảnh để nhận diện

 Thêm người mới qua giao diện

 Phân cụm để cải thiện nhận diện

 Triển khai trên Docker
