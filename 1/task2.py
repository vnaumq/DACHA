import cv2
import os
import numpy as np

# Определяем путь к папке files относительно расположения скрипта
base_dir = os.path.dirname(__file__)  # Директория, где находится скрипт
files_dir = os.path.join(base_dir, "files", "photo1")  # Путь к папке files
output_folder = os.path.join(base_dir, "files", "task2")  # Папка для сохранения результатов

# Целевой размер изображений
TARGET_SIZE = (450, 500)  # Ширина 450, высота 500

# Проверяем текущую рабочую директорию
print("Текущая рабочая директория:", os.getcwd())

# Проверяем, существует ли папка с изображениями
if not os.path.exists(files_dir):
    print(f"Папка {files_dir} не найдена.")
    exit()

# Создаем папку для сохранения, если её нет
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Папка {output_folder} создана.")

# Обрабатываем все изображения в папке
for filename in os.listdir(files_dir):
    if filename.lower().endswith((".jpg", ".png")):  # Обрабатываем только изображения
        image_path = os.path.join(files_dir, filename)
        print(f"Обработка изображения: {filename}")

        # Загружаем изображение
        img = cv2.imread(image_path)
        if img is None:
            print(f"Не удалось загрузить {filename}")
            continue

        # Изменяем размер изображения до 450x500
        resized_img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

        # Сохраняем изменённое изображение
        resized_path = os.path.join(output_folder, f"{filename}_resized.jpg")
        cv2.imwrite(resized_path, resized_img)

        # Определяем область для обрезки (пример: центр изображения, 200x200 пикселей)
        crop_size = (200, 200)  # Размер обрезаемой области (ширина, высота)
        x_center = TARGET_SIZE[0] // 2  # Центр по ширине
        y_center = TARGET_SIZE[1] // 2  # Центр по высоте
        x_start = max(0, x_center - crop_size[0] // 2)
        y_start = max(0, y_center - crop_size[1] // 2)
        x_end = min(TARGET_SIZE[0], x_start + crop_size[0])
        y_end = min(TARGET_SIZE[1], y_start + crop_size[1])

        # Обрезаем область
        cropped_area = resized_img[y_start:y_end, x_start:x_end]

        # Сохраняем обрезанную область
        cropped_path = os.path.join(output_folder, f"{filename}_cropped.jpg")
        cv2.imwrite(cropped_path, cropped_area)

        # Создаём изображение без обрезанной области (заменяем её чёрным)
        img_without_crop = resized_img.copy()
        img_without_crop[y_start:y_end, x_start:x_end] = 0  # Заполняем чёрным

        # Сохраняем изображение без обрезанной области
        without_crop_path = os.path.join(output_folder, f"{filename}_without_crop.jpg")
        cv2.imwrite(without_crop_path, img_without_crop)

        # Применяем размытие по Гауссу с разными радиусами
        radii = [5, 15, 25]  # Радиусы размытия
        for radius in radii:
            blurred = cv2.GaussianBlur(resized_img, (radius, radius), 0)
            cv2.imwrite(
                os.path.join(output_folder, f"{filename}_blurred_{radius}.jpg"), blurred
            )

        # Обнаруживаем края с разными порогами
        thresholds = [(100, 200), (50, 150), (150, 250)]  # Пороги для Canny
        for low, high in thresholds:
            edges = cv2.Canny(resized_img, low, high)
            cv2.imwrite(
                os.path.join(output_folder, f"{filename}_edges_{low}_{high}.jpg"), edges
            )

print("Обработка завершена! Результаты сохранены в папку 'task2'.")