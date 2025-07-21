import cv2
import torch
from ultralytics import YOLO

# test
'''
model_path = 'C:/pythonProject/Pycharm_pythoncode/FACE ID/yolov8n-face.pt'
model = YOLO(model_path)
CONF = 0.7
image = 'C:/pythonProject/Pycharm_pythoncode/Face detection/Dataset/val/images/5897.jpg'
results = model.predict(image, verbose=False, conf=CONF)[0]

print(results)
print(dir(results))
print("Boxes:", results.boxes)
print("Confidences:", results.boxes.conf)
print("XYXY:", results.boxes.xyxy)
print("Class IDs:", results.boxes.cls)

'''
class FaceDetector:
    def __init__(self, model_path, device = None, conf_thres = 0.26):
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        self.conf_thres = conf_thres

    def detect_faces(self, image, draw = False):
        # du doan cac bbox tren anh voi nguong loc 0.3
        result = self.model.predict(image, verbose = False, conf = self.conf_thres)[0]
        boxes = []
        confidences = []
        for box in result.boxes:
            x1, y1, x2 ,y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0]) # chuyen toa do tu tensor ve so nguyen

            boxes.append([x1, y1, x2, y2])
            confidences.append(conf)

            if draw:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(image, f'{conf:.2f}', (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        return boxes, confidences, image if draw else image

if __name__ == '__main__':
    detector = FaceDetector(model_path = 'C:/pythonProject/Pycharm_pythoncode/FACE ID/yolov8n-face.pt')
    img = cv2.imread('C:/pythonProject/Pycharm_pythoncode/FACE ID/face_bank/Park Seo Jun/seojun3.jpg')
    boxes, confs, img_with_boxes = detector.detect_faces(img, draw=True)
    # print(boxes)
    # for box in boxes:
    #     x1, y1, x2, y2 = map(int, box)
    # print(x1,y1,x2,y2)

    cv2.imshow('Faces', img_with_boxes)
    cv2.waitKey(0)


















