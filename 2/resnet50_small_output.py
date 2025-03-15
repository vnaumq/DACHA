import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, ResNet152
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Загрузка и подготовка данных
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Список классов
#class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Shirt/Coat', 'Bag', 'Ankle boot/Sandal/Sneaker']

# Нормализация изображений
x_train = x_train / 255.0
x_test = x_test / 255.0

# Добавляем 3 канала (RGB) и изменяем размер до 32x32
x_train = np.stack([x_train] * 3, axis=-1)  # (60000, 28, 28, 3)
x_test = np.stack([x_test] * 3, axis=-1)  # (10000, 28, 28, 3)

x_train = tf.image.resize(x_train, (32, 32))  # Resize до 32x32
x_test = tf.image.resize(x_test, (32, 32))  # Resize до 32x32

y_train_copy = y_train.copy()

for i in range(len(y_train)):
  if y_train_copy[i] == 5:
    y_train_copy[i] = 6
    continue
  if y_train_copy[i] == 6:
    y_train_copy[i] = 4
    continue
  if y_train_copy[i] == 7:
    y_train_copy[i] = 6
    continue
  if y_train_copy[i] == 8:
    y_train_copy[i] = 5
    continue
  if y_train_copy[i] == 9:
    y_train_copy[i] = 6
    continue

y_test_copy = y_test.copy()

for i in range(len(y_test)):
  if y_test_copy[i] == 5:
    y_test_copy[i] = 6
    continue
  if y_test_copy[i] == 6:
    y_test_copy[i] = 4
    continue
  if y_test_copy[i] == 7:
    y_test_copy[i] = 6
    continue
  if y_test_copy[i] == 8:
    y_test_copy[i] = 5
    continue
  if y_test_copy[i] == 9:
    y_test_copy[i] = 6
    continue

y_train = np.array(y_train_copy)
y_test = np.array(y_test_copy)

# Преобразование меток в one-hot encoding
y_train = keras.utils.to_categorical(y_train, 7)
y_test = keras.utils.to_categorical(y_test, 7)

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
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dense(96, activation='relu'),
    Dense(7, activation='softmax')
])

# Компиляция модели
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Вывод структуры модели
model.summary()

# Создаем callback EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Метрика, которую будем отслеживать (потери на валидации)
    patience=30,          # Количество эпох без улучшения, после которых обучение остановится
    restore_best_weights=True  # Восстановление весов модели с лучшей эпохи
)

if "fashion_trained_model_3.keras" not in os.listdir():
    history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test), callbacks=[early_stopping])
    model.save("fashion_trained_model_3.keras")

    # Построение графика точности на обучающей и тестовой выборках
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
else:
    # Если модель уже обучена, загружаем её
    model = keras.models.load_model('fashion_trained_model_3.keras')



# Оценка точности на тестовой выборке
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)
predictions = model.predict(x_test)



# Визуализация примеров изображений
plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i])
    # Предсказание классов
    predicted_class = np.argmax(predictions[i])
    true_class = np.argmax(y_test[i])
    plt.xlabel(f'Predicted: {class_names[predicted_class]}, Actual: {class_names[true_class]}')
plt.show()