# файл является cifar_trained_model.keras обученной моделью

import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Загрузка и подготовка данных
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Список классов
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Нормализация изображений
x_train = x_train / 255.0
x_test = x_test / 255.0

# Преобразование меток в one-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Визуализация примеров изображений
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i])
    plt.xlabel(class_names[np.argmax(y_train[i])])
plt.show()


# Загрузка предобученной модели ResNet50 без верхнего слоя
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
base_model.trainable = False  # Замораживаем веса

# Создание новой модели на основе ResNet50
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Вывод структуры модели
model.summary()

## Обучение модели
#model.fit(x_train, y_train, batch_size=32, epochs=100, validation_split=0.2)

## Сохранение модели
#model.save("cifar_trained_model.keras")

# Загрузка модели
model = keras.models.load_model('cifar_trained_model.keras')

# Оценка точности на тестовой выборке
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)
predictions = model.predict(x_test)


# Визуализация примеров изображений
plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    k = random.randint(0, 300)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[k])
    # Предсказание классов
    predicted_class = np.argmax(predictions[k])
    true_class = np.argmax(y_test[k])
    plt.xlabel(f'Predicted: {class_names[predicted_class]}, Actual: {class_names[true_class]}')
plt.show()
