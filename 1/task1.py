import cv2
import numpy as np
import os

# Определяем путь к папке files относительно расположения скрипта
base_dir = os.path.dirname(__file__)
files_dir = os.path.join(base_dir, "files", "photo1")

# Максимальный размер изображения (в пикселях)
MAX_SIZE = (800, 800)  # Ограничение 800x800 пикселей

# Получаем все файлы .jpg из папки и берём первые 5
image_files = [f for f in os.listdir(files_dir) if f.lower().endswith('.jpg')]
image_paths = [os.path.join(files_dir, f) for f in image_files[:5]]

# Параметры сетки
rows, cols = 2, 3
grid_images = []

# Чтение и обработка изображений
for i, path in enumerate(image_paths):
    try:
        # Читаем изображение через OpenCV (BGR формат)
        img = cv2.imread(path)
        if img is None:
            raise Exception("Не удалось загрузить изображение")

        # Получаем размеры изображения
        h, w = img.shape[:2]

        # Проверяем и уменьшаем размер если нужно
        if w > MAX_SIZE[0] or h > MAX_SIZE[1]:
            scale = min(MAX_SIZE[0]/w, MAX_SIZE[1]/h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Добавляем название
        filename = os.path.basename(path)[:15] + "..."
        cv2.putText(img, f"Img {i+1}: {filename}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)

        # Сохраняем обработанное изображение
        grid_images.append(img)

    except Exception as e:
        print(f"Ошибка при обработке {path}: {str(e)}")
        # Добавляем пустое изображение в случае ошибки
        grid_images.append(np.zeros((MAX_SIZE[1], MAX_SIZE[0], 3), dtype=np.uint8))

# Создаем пустую сетку
if grid_images:
    # Определяем максимальные размеры для ячеек сетки
    max_h = max(img.shape[0] for img in grid_images)
    max_w = max(img.shape[1] for img in grid_images)

    # Создаем пустое изображение для всей сетки
    grid_height = max_h * rows
    grid_width = max_w * cols
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    # Заполняем сетку изображениями с центрированием
    for idx, img in enumerate(grid_images):
        row = idx // cols
        col = idx % cols

        # Размеры текущего изображения
        h, w = img.shape[:2]

        # Вычисляем смещение для центрирования
        y_offset = (max_h - h) // 2  # Вертикальное смещение
        x_offset = (max_w - w) // 2  # Горизонтальное смещение

        # Координаты в сетке
        y_start = row * max_h + y_offset
        x_start = col * max_w + x_offset

        # Вставляем изображение с учётом смещения
        grid[y_start:y_start+h, x_start:x_start+w] = img

    # Конвертируем BGR в RGB для корректного отображения
    grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)

    # Показываем результат
    cv2.imshow("Image Grid", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Нет изображений для отображения")