import os
import shutil
from sklearn.model_selection import train_test_split

src_root = 'C:/pythonProject/Pycharm_pythoncode/FACE ID/lfw-funneled/lfw_funneled'
dst_root = 'C:/pythonProject/Pycharm_pythoncode/FACE ID/Dataset_1'

train_dir = os.path.join(dst_root,'train')
val_dir = os.path.join(dst_root,'val')

for person_folder in sorted(os.listdir(src_root)):
    person_path = os.path.join(src_root, person_folder)
    if not os.path.isdir(person_path):
        continue

    # lay danh sach anh
    image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    # phan chia
    train_imgs, val_imgs = train_test_split(image_files, test_size=0.2, random_state=42)

    #tao thu muc va sao chep
    train_person_dir = os.path.join(train_dir, person_folder)
    val_person_dir = os.path.join(val_dir,person_folder)

    os.makedirs(train_person_dir, exist_ok=True)
    os.makedirs(val_person_dir,exist_ok=True)

    for img in train_imgs:
        shutil.copy2(os.path.join(person_path, img), os.path.join(train_person_dir,img))

    for img in val_imgs:
        shutil.copy2(os.path.join(person_path, img), os.path.join(val_person_dir,img))
    