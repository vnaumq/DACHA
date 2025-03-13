import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

# Загрузка и подготовка данных
def load_and_prepare_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Нормализация изображений
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Добавляем 3 канала (RGB) и изменяем размер до 32x32
    x_train = np.stack([x_train] * 3, axis=-1)  # (60000, 28, 28, 3)
    x_test = np.stack([x_test] * 3, axis=-1)  # (10000, 28, 28, 3)
    
    x_train = tf.image.resize(x_train, (32, 32))  # Resize до 32x32
    x_test = tf.image.resize(x_test, (32, 32))  # Resize до 32x32
    
    # Преобразование меток в one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return x_train, y_train, x_test, y_test

# Визуализация примеров изображений
def visualize_images(images, labels, class_names, num_images=25):
    plt.figure(figsize=(10,10))
    for i in range(num_images):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i])
        plt.xlabel(class_names[np.argmax(labels[i])])
    plt.show()

# Создание модели
def create_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    base_model.trainable = False  # Замораживаем веса
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Обучение модели
def train_model(model, x_train, y_train, x_test, y_test, epochs=20, batch_size=32):
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    model.save("fashion_trained_model_2.keras")
    return history

# Построение графика точности
def plot_accuracy(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Оценка модели на тестовой выборке
def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)
    return model.predict(x_test)

# Визуализация предсказаний
def visualize_predictions(images, predictions, true_labels, class_names, num_images=9):
    plt.figure(figsize=(10,10))
    for i in range(num_images):
        plt.subplot(3,3,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i])
        predicted_class = np.argmax(predictions[i])
        true_class = np.argmax(true_labels[i])
        plt.xlabel(f'Predicted: {class_names[predicted_class]}, Actual: {class_names[true_class]}')
    plt.show()

# Главная функция
def main():
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    x_train, y_train, x_test, y_test = load_and_prepare_data()
    visualize_images(x_train, y_train, class_names)
    
    model = create_model()
    model.summary()
    
    if "fashion_trained_model_2.keras" not in os.listdir():
        history = train_model(model, x_train, y_train, x_test, y_test)
        plot_accuracy(history)
    else:
        model = keras.models.load_model('fashion_trained_model_2.keras')
    
    predictions = evaluate_model(model, x_test, y_test)
    visualize_predictions(x_test, predictions, y_test, class_names)

if __name__ == "__main__":
    main()