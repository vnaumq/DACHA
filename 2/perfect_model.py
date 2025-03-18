# Импорт необходимых библиотек
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2
import os
from tensorflow.keras.callbacks import EarlyStopping

# Константы для улучшения читаемости и гибкости кода
IMG_SHAPE = (28, 28, 1)
NUM_CLASSES = 7
BATCH_SIZE = 128
EPOCHS = 45
VALIDATION_SPLIT = 0.2
MODEL_PATH = 'perfect_model.keras'


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

    # Словарь для замены значений классов
    mapping = {5: 6, 6: 4, 7: 6, 8: 5, 9: 6}

    # Применение замены с помощью numpy
    y_train = np.array([mapping.get(y, y) for y in y_train])
    y_test = np.array([mapping.get(y, y) for y in y_test])

    return (x_train, y_train), (x_test, y_test)

def create_model():
    """Создание архитектуры сверточной нейронной сети"""
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=IMG_SHAPE, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.5),
        Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

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


    # Проверка существования сохраненной модели
    if not os.path.exists(MODEL_PATH):
        early_stopping = EarlyStopping(
          monitor='val_loss',  # Метрика, которую будем отслеживать (потери на валидации)
          patience=10,          # Количество эпох без улучшения, после которых обучение остановится
          restore_best_weights=True  # Восстановление весов модели с лучшей эпохи
      )

        # Создание и обучение новой модели
        model = create_model()
        history = model.fit(
            x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            shuffle=True,
            validation_split=VALIDATION_SPLIT,
            callbacks = [early_stopping]
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