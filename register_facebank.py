import os
import cv2
import pickle
import numpy as np
from sklearn.cluster import KMeans
from Face_Detector import FaceDetector
from Face_Embedder import FaceEmbedder

# ===== Cấu hình =====
face_dir = "C:/pythonProject/Pycharm_pythoncode/FACE ID/face_bank"
save_path = "C:/pythonProject/Pycharm_pythoncode/FACE ID/facebank.pkl"
n_clusters = 2  # Số cụm đặc trưng cho mỗi người

# ===== Khởi tạo model =====
detector = FaceDetector(model_path='C:/pythonProject/Pycharm_pythoncode/FACE ID/yolov8n-face.pt')
embedder = FaceEmbedder(model_path='C:/pythonProject/Pycharm_pythoncode/FACE ID/best_model.pt')

facebank = []

def cluster_embeddings(embeddings, n_clusters=2):
    """Phân cụm embeddings và trả về tâm cụm"""
    if len(embeddings) < n_clusters:
        # Nếu không đủ ảnh để phân cụm thì trả về 1 vector trung bình
        return [np.mean(embeddings, axis=0)]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    return list(kmeans.cluster_centers_)

# ===== Xử lý từng người =====
for person_name in os.listdir(face_dir):
    person_path = os.path.join(face_dir, person_name)
    if not os.path.isdir(person_path):
        continue

    embeddings = []

    for filename in os.listdir(person_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(person_path, filename)
            img = cv2.imread(img_path)

            boxes = detector.detect_faces(img)
            if not boxes:
                print(f" Không phát hiện khuôn mặt trong ảnh: {filename}")
                continue

            try:
                x1, y1, x2, y2 = map(int, boxes[0][0])  # chỉ lấy khuôn mặt đầu tiên
                face_crop = img[y1:y2, x1:x2]
                embedding = embedder.get_embedding(face_crop)

                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    print(f" Không tạo được embedding cho {filename}")

            except Exception as e:
                print(f" Lỗi xử lý ảnh {filename}: {e}")

    if embeddings:
        centers = cluster_embeddings(np.array(embeddings), n_clusters=n_clusters)
        for i, center in enumerate(centers):
            name_id = f"{person_name}_{i}" if len(centers) > 1 else person_name
            facebank.append((name_id, center))
        print(f" Đã thêm {len(centers)} vector đặc trưng cho {person_name}")

    else:
        print(f" Không có ảnh hợp lệ cho {person_name}")

# ===== Lưu facebank =====
with open(save_path, 'wb') as f:
    pickle.dump(facebank, f)

print(f" Đã lưu FaceBank vào: {save_path}")
