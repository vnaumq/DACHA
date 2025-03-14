import cv2
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Путь к папке с изображениями
folder_path = BASE_DIR + '/files/ten_clothing_images'

# Список файлов в папке
image_files = os.listdir(folder_path)

# Создание папки для сохранения обработанных изображений
output_folder = BASE_DIR+ '/files/ten_clothing_images_resized'
os.makedirs(output_folder, exist_ok=True)

# Обработка каждого изображения
for file_name in image_files:
    # Полный путь к файлу
    file_path = os.path.join(folder_path, file_name)

    # Загрузка изображения
    img = cv2.imread(file_path)
    if img is None:
        print(f"Не удалось загрузить {file_name}")
        continue
    else:
        print("NICE")

    # Преобразуем изображение в черно-белое
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Изменяем размер до 280x280
    img_resized = cv2.resize(img_gray, (280, 280))

    # Сохраняем изображение
    save_path = os.path.join(output_folder, file_name)
    cv2.imwrite(save_path, img_resized)