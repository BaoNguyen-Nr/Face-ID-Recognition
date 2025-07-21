# 1. Base image có Python
FROM python:3.11-slim

# 2. Thiết lập thư mục làm việc trong container
WORKDIR /app

# 3. Copy toàn bộ code từ thư mục hiện tại vào container
COPY . /app

# 4. Cài đặt các thư viện cần thiết
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 5. Cài thêm các thư viện hệ thống cần thiết (OpenCV yêu cầu)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# 6. Expose port để chạy Streamlit
EXPOSE 8501

# 7. Command để chạy app Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
