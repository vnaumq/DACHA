# Импорт необходимых библиотек
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
import os

# Константы для улучшения читаемости и гибкости кода
IMG_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 15
VALIDATION_SPLIT = 0.2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'files', 'model', 'perfect_model.keras')

def load_and_preprocess_data():
    """Загрузка и предобработка данных Fashion MNIST"""
    # Загрузка набора данных
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Изменение формы данных для сверточной сети (добавление канала)
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))

    # Нормализация значений пикселей в диапазон [0, 1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return (x_train, y_train), (x_test, y_test)

def create_model():
    """Создание архитектуры сверточной нейронной сети"""
    model = Sequential([
        # Первый сверточный слой: 32 фильтра размером 3x3
        Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SHAPE),
        # Второй сверточный слой: 64 фильтра
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        # Слой субдискретизации для уменьшения размерности
        MaxPooling2D((2, 2)),
        # Dropout для предотвращения переобучения
        Dropout(0.5),
        # Третий сверточный слой: 128 фильтров
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.5),
        # Преобразование в одномерный вектор
        Flatten(),
        # Полносвязный слой с 128 нейронами
        Dense(128, activation='relu'),
        # Выходной слой с softmax для классификации
        Dense(NUM_CLASSES, activation='softmax')
    ])

    # Компиляция модели
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def plot_training_history(history):
    """Визуализация истории обучения"""
    plt.figure(figsize=(12, 5))

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Обучающая выборка')
    plt.plot(history.history['val_loss'], label='Валидационная выборка')
    plt.title('Функция потерь по эпохам')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.grid(True)

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Обучающая выборка')
    plt.plot(history.history['val_accuracy'], label='Валидационная выборка')
    plt.title('Точность по эпохам')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    """Основная функция программы"""
    # Загрузка и предобработка данных
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    # Создание директории для модели, если она не существует
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Проверка существования сохраненной модели
    if not os.path.exists(MODEL_PATH):
        # Создание и обучение новой модели
        model = create_model()
        history = model.fit(
            x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            shuffle=True,
            validation_split=VALIDATION_SPLIT,
        )
        # Сохранение обученной модели
        model.save(MODEL_PATH)
    else:
        # Загрузка существующей модели
        model = keras.models.load_model(MODEL_PATH)
        history = None

    # Оценка модели на тестовых данных
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Точность на тестовых данных: {test_acc:.4f}')

    # Визуализация результатов обучения, если модель обучалась
    if history:
        plot_training_history(history)
    else:
        print('Модель загружена из файла')

if __name__ == '__main__':
    main()