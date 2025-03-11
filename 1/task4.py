import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl

# Определяем путь к папке files относительно расположения скрипта
base_dir = os.path.dirname(__file__)  # Директория, где находится скрипт
files_dir = os.path.join(base_dir, "files1")  # Путь к папке files
output_folder = os.path.join(
    base_dir, "processed2_files"
)  # Папка для сохранения результатов


# Функция для вычисления гистограмм и статистики
def process_image(image_path, df):
    # Загружаем изображение
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: Не удалось загрузить изображение {image_path}")
        return df

    # Переводим изображение в цветовую модель HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Разделяем каналы H, S, V
    h, s, v = cv2.split(hsv_img)

    # Вычисляем гистограммы для каждого канала
    hist_h = cv2.calcHist([h], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
    hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])

    # Нормализуем гистограммы
    hist_h = hist_h / hist_h.sum()
    hist_s = hist_s / hist_s.sum()
    hist_v = hist_v / hist_v.sum()

    # Сохраняем гистограммы в DataFrame
    temp_df = pd.DataFrame(
        {
            "Image": os.path.basename(image_path),
            "Channel": ["H"] * 256 + ["S"] * 256 + ["V"] * 256,
            "Bin": list(range(256)) * 3,
            "Value": np.concatenate(
                [hist_h.flatten(), hist_s.flatten(), hist_v.flatten()]
            ),
        }
    )
    df = pd.concat([df, temp_df], ignore_index=True)

    return df


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

# Создаем пустой DataFrame для хранения гистограмм
df = pd.DataFrame(columns=["Image", "Channel", "Bin", "Value"])

# Обрабатываем все изображения в папке
for filename in os.listdir(files_dir):
    if filename.endswith(".jpg") or filename.endswith(
        ".png"
    ):  # Обрабатываем только изображения
        image_path = os.path.join(files_dir, filename)
        print(f"Обработка изображения: {filename}")
        df = process_image(image_path, df)

# Вычисляем среднее, медиану и стандартное отклонение для каждого канала
statistics = (
    df.groupby(["Image", "Channel"])["Value"]
    .agg(["mean", "median", "std"])
    .reset_index()
)

# Сохраняем статистику в CSV
statistics.to_excel(os.path.join(output_folder, "statistics.xlsx"), index=False)

# Визуализация гистограмм для сравнения
for channel in ["H", "S", "V"]:
    plt.figure(figsize=(10, 6))
    for image in df["Image"].unique():
        channel_data = df[(df["Image"] == image) & (df["Channel"] == channel)]
        plt.plot(channel_data["Bin"], channel_data["Value"], label=image)
    plt.title(f"Гистограмма канала {channel}")
    plt.xlabel("Бин")
    plt.ylabel("Значение")
    plt.legend()
    plt.savefig(os.path.join(output_folder, f"histogram_{channel}.png"))
    plt.show()

print("Обработка завершена! Результаты сохранены в папку 'processed1_files'.")
