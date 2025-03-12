# Импорт библиотек
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

# Путь к файлу модели
MODEL_PATH = 'fashion_mnist_model.h5'

# Загрузка данных Fashion MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))
# Нормализация данных
x_train = x_train / 255.0
x_test = x_test / 255.0

# Проверка, существует ли модель
if os.path.exists(MODEL_PATH):
    print("Загрузка существующей модели...")
    model = load_model(MODEL_PATH)
else:
    print("Создание новой модели...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.5),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Компиляция модели
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Обучение модели
    history = model.fit(x_train, y_train, epochs=15, batch_size=512, shuffle=True, validation_split=0.1)

    # Сохранение модели
    model.save(MODEL_PATH)
    print("Модель сохранена в файл:", MODEL_PATH)

    # Оценка модели
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)

    # Построение графиков
    plt.figure(figsize=(12, 5))

    # 1. График функции потерь
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. График точности
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Отображение графиков
    plt.tight_layout()
    plt.show()

# Функция для загрузки и предобработки изображений
def load_and_preprocess_images(folder_path):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Загрузка в градациях серого
        img = cv2.resize(img, (28, 28))  # Изменение размера до 28x28
        img = img.reshape((28, 28, 1))  # Добавление размерности канала
        img = img / 255.0  # Нормализация
        images.append(img)
        filenames.append(filename)  # Сохраняем имя файла
    return np.array(images), filenames

# Загрузка и предобработка изображений из папки
images, filenames = load_and_preprocess_images('files/ten_clothing_images_resized/inv')

# Классификация изображений
predictions = model.predict(images)

# Вывод результатов
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

for i, (pred, filename) in enumerate(zip(predictions, filenames)):
    print(pred)
    print(pred.max())
    predicted_class = np.argmax(pred)  # Индекс класса с максимальной вероятностью
    confidence = pred[predicted_class]  # Уверенность модели
    print(f"File: {filename}, Predicted class: {class_names[predicted_class]}, Confidence: {confidence:.2f}")
    print("="*100)