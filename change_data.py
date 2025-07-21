import os
import cv2
import shutil

path = 'C:/pythonProject/Pycharm_pythoncode/FACE ID/Local_FacesDataset/Tst'
dst_roof = 'C:/pythonProject/Pycharm_pythoncode/FACE ID/Dataset'
# tao thu muc
os.makedirs(dst_roof, exist_ok=True)

# doi ten

for idx, person_folder in enumerate(sorted(os.listdir(path))):
    person_path = os.path.join(path, person_folder)
    if os.path.isdir(person_path):
        new_person_dir = os.path.join(dst_roof,f'person_{idx+1}')
        os.makedirs(new_person_dir, exist_ok=True)
        for j,img_file in enumerate(os.listdir(person_path)):# truy cap vao buc anh con
            src_img_path = os.path.join(person_path,img_file)
            dst_img_path = os.path.join(new_person_dir, f'person_{idx+1}_{j+1}.jpg')
            shutil.copy2(src_img_path,dst_img_path)
