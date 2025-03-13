import cv2
import numpy as np
import os

# Определяем путь к папке files относительно расположения скрипта
base_dir = os.path.dirname(__file__)
files_dir = os.path.join(base_dir, "files", "task1")

# Получаем список всех JPG-файлов и берем первые 5
image_names = [f for f in os.listdir(files_dir) if f.lower().endswith('.jpg')]
image_names = image_names[:5]

# Полные пути к изображениям
image_paths = [os.path.join(files_dir, name) for name in image_names]

# Загружаем изображения
images = []
for path in image_paths:
    try:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    except FileNotFoundError:
        print(f"Файл не найден: {path}")
        images.append(None)

# Проверяем, достаточно ли изображений
if len(images) < 5:
    print(f"Найдено только {len(images)} изображений из требуемых 5")
    # Заполняем недостающие изображения черными прямоугольниками позже

# Определяем максимальные размеры только для существующих изображений
valid_images = [img for img in images if img is not None]
if not valid_images:
    print("Нет доступных изображений для обработки")
    exit()

max_height = max(img.shape[0] for img in valid_images)
max_width = max(img.shape[1] for img in valid_images)

# Приводим все изображения к одному размеру и заполняем недостающие
for i in range(5):
    if i < len(images) and images[i] is not None:
        # Изменяем размер изображения, если он отличается
        images[i] = cv2.resize(images[i], (max_width, max_height))
    else:
        # Создаем черное изображение нужного размера
        images.append(np.zeros((max_height, max_width, 3), dtype=np.uint8))

# Создаем сетку 2x3
try:
    row1 = np.hstack((images[0], images[1], images[2]))
    row2 = np.hstack((images[3], images[4], np.zeros((max_height, max_width, 3), dtype=np.uint8)))
    grid = np.vstack((row1, row2))

    # Добавляем заголовки
    for i in range(5):
        if images[i].sum() > 0:  # Проверяем, не черное ли изображение
            row = 0 if i < 3 else 1
            col = i % 3
            x = col * max_width + 10
            y = row * max_height + 30
            cv2.putText(grid, f"Image {i+1}", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Отображаем результат
    cv2.imshow("Image Grid", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except ValueError as e:
    print(f"Ошибка при объединении изображений: {e}")
    print("Проверьте, что все изображения имеют одинаковую высоту")