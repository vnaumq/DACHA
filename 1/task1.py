import matplotlib.pyplot as plt
from PIL import Image
import os

# Определяем путь к папке files относительно расположения скрипта
base_dir = os.path.dirname(__file__)  # Директория, где находится скрипт
files_dir = os.path.join(base_dir, "files")  # Путь к папке files

# Пути к изображениям
image_names = [
    "photo_2025-03-10_20-21-20.jpg",
    "photo_2025-03-10_20-21-25.jpg",
    "photo_2025-03-10_20-21-27.jpg",
    "photo_2025-03-10_20-21-30.jpg",
    "photo_2025-03-10_20-21-33.jpg",
]

# Полные пути к изображениям
image_paths = [os.path.join(files_dir, name) for name in image_names]

# Создание сетки 2x3 (2 строки, 3 столбца)
plt.figure(figsize=(10, 6))  # Размер фигуры

for i, path in enumerate(image_paths, 1):
    try:
        img = Image.open(path)  # Открываем изображение
        plt.subplot(2, 3, i)  # 2 строки, 3 столбца, i-я позиция
        plt.imshow(img)
        plt.title(f"Image {i}")  # Заголовок для изображения
        plt.axis("off")  # Отключение осей
    except FileNotFoundError:
        print(f"Файл не найден: {path}")

plt.tight_layout()  # Автоматическая настройка расстояний между изображениями
plt.show()  # Отображение сетки с изображениями
