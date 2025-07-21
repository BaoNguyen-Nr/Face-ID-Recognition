import cv2
from recognize_face import FaceRecognizer

# Đường dẫn ảnh test
image_path = 'C:/pythonProject/Pycharm_pythoncode/FACE ID/test.jpeg'

# Load ảnh BGR
image_bgr = cv2.imread(image_path)

# Check ảnh có tồn tại không
if image_bgr is None:
    raise FileNotFoundError(f"[!] Không thể đọc được ảnh từ: {image_path}")

# Khởi tạo FaceRecognizer
recognizer = FaceRecognizer()

# Nhận diện khuôn mặt
bboxes, names, annotated_img = recognizer.recognize_faces(image_bgr, draw=True)

# In kết quả
print(f"[INFO] Phát hiện {len(bboxes)} khuôn mặt:")
for name, bbox in zip(names, bboxes):
    print(f" - Tên: {name} | Vị trí: {bbox}")

# Hiển thị ảnh với kết quả
cv2.imshow("Face Recognition Result", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
