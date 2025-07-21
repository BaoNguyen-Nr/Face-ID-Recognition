import os
import cv2
import pickle
import numpy as np
from Face_Detector import FaceDetector
from Face_Embedder import FaceEmbedder

Model_Path = 'C:/pythonProject/Pycharm_pythoncode/FACE ID/best_model.pt'
FBK = 'C:/pythonProject/Pycharm_pythoncode/FACE ID/facebank.pkl'

class FaceRecognizer:
    def __init__(self, model_path=Model_Path, facebank_path=FBK, threshold=0.5):
        self.detector = FaceDetector(model_path='C:/pythonProject/Pycharm_pythoncode/FACE ID/yolov8n-face.pt')
        self.embedder = FaceEmbedder(model_path)
        self.threshold = threshold

        # Load facebank
        if os.path.exists(facebank_path):
            with open(facebank_path, 'rb') as f:
                raw_facebank = pickle.load(f)

            # Xử lý facebank: mỗi người có thể có nhiều vector (cụm), nên luôn là dict[str -> list[np.array]]
            self.facebank = {}
            if isinstance(raw_facebank, dict):
                self.facebank = raw_facebank
            elif isinstance(raw_facebank, list):  # fallback cho định dạng cũ
                for name, vector in raw_facebank:
                    self.facebank[name] = [vector]
            else:
                raise ValueError("Định dạng facebank không hợp lệ.")
        else:
            raise FileNotFoundError(f"Facebank not found: {facebank_path}")

    def cosine_similarity(self, a, b):
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        return np.dot(a, b)

    def recognize_faces(self, image_bgr, draw=True, min_face_size=40):
        boxes, confs, img_with_boxes = self.detector.detect_faces(image_bgr)
        names = []
        face_locations = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            if w < min_face_size or h < min_face_size:
                continue

            face_crop = image_bgr[y1:y2, x1:x2]
            emb = self.embedder.get_embedding(face_crop)

            if emb is None:
                names.append("Unknown")
                face_locations.append(box)
                continue

            best_sim = -1
            identity = "Unknown"

            # So sánh với từng người
            for name, vectors in self.facebank.items():
                sims = [self.cosine_similarity(emb, vec) for vec in vectors]
                max_sim = max(sims)

                if max_sim > best_sim:
                    best_sim = max_sim
                    identity = name

            if best_sim < self.threshold:
                identity = "Unknown"

            names.append(identity)
            face_locations.append(box)

            if draw:
                color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
                label = f'{identity} ({best_sim:.2f})' if identity != "Unknown" else identity
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image_bgr, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

        return face_locations, names, image_bgr
