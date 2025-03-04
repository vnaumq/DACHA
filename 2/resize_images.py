import cv2
import os 
import numpy as np 

# путь к папке
folder_path = 'files/ten_clothing_images'

# Список файлов в папке
image_files = os.listdir(folder_path)

# Создание папки для сохранения обработанных картинок 
outher_folder = 'files/ten_clothing_images_resized'
os.makedirs(outher_folder, exist_ok=True)

# Обработка каждлго изображения
for file_name in image_files:
    # полный путь к файлу
    file_path = os.path.join(folder_path, file_name)
    
    # Загрузка изображения 
    img = cv2.imread(file_path)
    if img is None:
        print(f"Не удалось загрузить {file_name}")
        continue
    else:
        print("NICE")
    
    # Изменяем размер до 280x280
    img_resized = cv2.resize(img, (280, 280))
    
    # сохраняем
    save_path = os.path.join(outher_folder, file_name)
    cv2.imwrite(save_path, img_resized)

