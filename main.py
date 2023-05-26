import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16


train_dir = '/Users/M S I/PycharmProjects/TugasDL1/Training'
test_dir = '/Users/M S I/PycharmProjects/TugasDL1/Testing'


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


test_datagen = ImageDataGenerator(rescale=1./255)


batch_size = 32
image_size = (224, 224)  # Ukuran input yang diharapkan oleh VGG16


train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'  # Menggunakan class_mode binary karena hanya ada 2 kelas (mobil dan motor)
)


test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

model = Sequential()

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(image_size[0], image_size[1], 3)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


model.add(Flatten())


model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Menggunakan sigmoid karena hanya ada 2 kelas (mobil dan motor)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


steps_per_epoch = train_data.samples // batch_size


history = model.fit(
    train_data,
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_data
)


test_loss, test_acc = model.evaluate(test_data)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)


class_names = list(train_data.class_indices.keys())


images, labels = next(test_data)


normalized_images = test_datagen.standardize(images)


predictions = model.predict(normalized_images)


fig, axes = plt.subplots(len(images), 2, figsize=(8, 8*len(images)))

for i, (image, prediction) in enumerate(zip(images, predictions)):
    
    axes[i, 0].imshow(image)
    axes[i, 0].axis('off')

    
    probabilities = tf.nn.sigmoid(prediction)
    predicted_class_indices = tf.where(probabilities > 0.5)[:, 0]
    predicted_class_names = [class_names[i] for i in predicted_class_indices]

   
    axes[i, 1].bar(range(len(class_names)), probabilities)
    axes[i, 1].set_xticks(range(len(class_names)))
    axes[i, 1].set_xticklabels(class_names, rotation=45)
    axes[i, 1].set_ylim([0, 1])
    axes[i, 1].set_title(f'Predicted: {predicted_class_names}')

plt.tight_layout()
plt.show()
