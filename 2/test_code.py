# Импорт библиотек
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

# импорт дата сета
from tensorflow.keras.datasets import fashion_mnist

# класс для создания линейной модели нейронной сети.
from tensorflow.keras.models import Sequential

# классы для добавления полносвязных слоев и слоев регуляризации в модель нейронной сети.
from tensorflow.keras.layers import Dense, Dropout

# вспомогательные функции
from tensorflow.keras import utils

# разделение на тренировочную и тестовую выборки
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# список названий классов
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# отображает первое изображение из тренировочного набора на цветовой шкале
plt.figure()
plt.imshow(x_train[1])
plt.colorbar()
plt.grid(False)
plt.show()

# Нормализует данные, деля каждое значение пикселя на 255, чтобы привести их в диапазон от 0 до 1.
x_train = x_train / 255
x_test = x_test / 255

plt.figure()
plt.imshow(x_train[1])
plt.colorbar()
plt.grid(False)
plt.show()

# первые 25 изображений из тренировочного набора в виде сетки 5x5
plt.figure(figsize=(10,10))
for i in range (25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(x_train[i], cmap=plt.cm.binary)
  plt.xlabel(class_names[y_train[i]])
  plt.show()
# Создает последовательную модель нейронной сети с тремя слоями:
# слой выравнивания (преобразует 2D-изображения в 1D-вектор),
# плотный слой с 128 нейронами и функцией активации ReLU и
# выходной слой с 10 нейронами и функцией активации softmax для предсказания классов.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10,activation="softmax")
])

# Компилирует модель, указывая оптимизатор стохастического градиентного спуска (SGD),
# функцию потерь (sparse categorical crossentropy) и метрику точности.
model.compile(optimizer=tf.keras.optimizers.SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Выводит сводку модели, показывая структуру модели, количество параметров и другие детали.
model.summary()

# Тренирует модель на тренировочных данных в течение 10 эпох.
model.fit(x_train, y_train, epochs=10)

# Оценивает модель на тестовых данных и возвращает потери и точность на тестовом наборе.
test_loss, test_acc = model.evaluate(x_test, y_test)

# Выводит точность модели на тестовом наборе данных.
print('Test accuracy:', test_acc)

# Делает предсказания для тренировочных данных, возвращая вероятности для каждого класса.
predictions = model.predict(x_train)

# Находит индекс класса с наибольшей вероятностью для 125-го изображения в тренировочном наборе.
np.argmax(predictions[124])

# Выводит реальную метку класса для 125-го изображения в тренировочном наборе.
print(f"y_train = {y_train[124]}")

plt.figure()
plt.imshow(x_train[124])
plt.colorbar()
plt.grid(False)
plt.show()

# Выводит название класса с наибольшей вероятностью для 125-го изображения на основе предсказаний модели.
print(f'class_names = {class_names[np.argmax(predictions[124])]}')
