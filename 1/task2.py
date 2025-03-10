import cv2
import os

# Определяем путь к папке files относительно расположения скрипта
base_dir = os.path.dirname(__file__)  # Директория, где находится скрипт
files_dir = os.path.join(base_dir, "files")  # Путь к папке files
output_folder = os.path.join(
    base_dir, "processed1_files"
)  # Папка для сохранения результатов

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
    if filename.endswith(".jpg") or filename.endswith(
        ".png"
    ):  # Обрабатываем только изображения
        image_path = os.path.join(files_dir, filename)
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

print("Обработка завершена! Результаты сохранены в папку 'processed1_files'.")
