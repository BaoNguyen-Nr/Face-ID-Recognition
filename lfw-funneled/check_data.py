import os
import shutil

# Đường dẫn thư mục gốc
path = 'C:/pythonProject/Pycharm_pythoncode/FACE ID/lfw-funneled/lfw_funneled'

# Biến thống kê
total_images_deleted = 0
total_images_remaining = 0
total_people_remaining = 0

# Duyệt qua từng thư mục con
for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)

    if os.path.isdir(folder_path):
        # Lấy danh sách các file ảnh (không bao gồm thư mục con)
        img_files = [f for f in os.listdir(folder_path)
                     if os.path.isfile(os.path.join(folder_path, f))]

        num_images = len(img_files)

        if num_images < 8:
            print(f"🗑️ Đang xóa thư mục: {folder} (chỉ có {num_images} ảnh)")
            total_images_deleted += num_images
            shutil.rmtree(folder_path)
        else:
            total_people_remaining += 1
            total_images_remaining += num_images

# In kết quả thống kê
print("\n📊 Thống kê sau khi lọc:")
print(f"🔴 Tổng số ảnh đã xóa: {total_images_deleted}")
print(f"🟢 Tổng số ảnh còn lại: {total_images_remaining}")
print(f"👤 Tổng số người còn lại: {total_people_remaining}")
