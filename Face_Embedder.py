import torch
import torch.nn.functional as F
import numpy as np
import cv2
from Model_Resnet import FaceNetEmbedding

class FaceEmbedder:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = FaceNetEmbedding(embedding_size=128).to(self.device)  # hoặc số khác nếu bạn đặt khác
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Nếu mô hình bạn cần normalization ảnh
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    def preprocess(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (112, 112))  # hoặc (160, 160) tùy theo mô hình bạn train
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0  # [C, H, W]
        img_tensor = (img_tensor - torch.tensor(self.mean)[:, None, None]) / torch.tensor(self.std)[:, None, None]
        return img_tensor.unsqueeze(0).to(self.device)  # [1, C, H, W]

    def get_embedding(self, face_img_bgr):
        try:
            face_tensor = self.preprocess(face_img_bgr)
            with torch.no_grad():
                embedding = self.model(face_tensor)  # [1, 128] hoặc bao nhiêu tùy bạn
                embedding = F.normalize(embedding, p=2, dim=1)  # chuẩn hóa L2
            return embedding.squeeze(0).cpu().numpy()
        except Exception as e:
            print(f"[!] Lỗi embedding: {e}")
            return None

    def compare_faces(self, known_embeddings, unknown_embedding, threshold=0.6):
        known_embeddings = np.array(known_embeddings)
        dists = np.linalg.norm(known_embeddings - unknown_embedding, axis=1)
        results = dists < threshold
        return results.tolist(), dists.tolist()
