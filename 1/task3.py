import cv2
import os

# Загружаем изображение
input_folder = "files/photo1"  # Папка с исходными изображениями
output_folder = "files/task3"  # Папка для сохранения результатов

# Создаем папку для сохранения, если её нет
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Обрабатываем все изображения в папке
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(
        ".png"
    ):  # Обрабатываем только изображения
        image_path = os.path.join(input_folder, filename)
        print(f"Обработка изображения: {filename}")

        # Загружаем изображение
        img = cv2.imread(image_path)

        # Применяем размытие по Гауссу с разными радиусами
        radii = [5, 15, 25]  # Радиусы размытия
        for radius in radii:
            blurred = cv2.GaussianBlur(img, (radius, radius), 0)
            cv2.imwrite(
                os.path.join(output_folder, f"{filename}_blurred_{radius}.jpg"), blurred
            )

        # Обнаруживаем края с разными порогами
        thresholds = [(100, 200), (50, 150), (150, 250)]  # Пороги для Canny
        for low, high in thresholds:
            edges = cv2.Canny(img, low, high)
            cv2.imwrite(
                os.path.join(output_folder, f"{filename}_edges_{low}_{high}.jpg"), edges
            )

print("Обработка завершена! Результаты сохранены в папку 'task3'.")
