import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score

# Mengatur seed agar hasil dapat direproduksi
np.random.seed(42)
tf.random.set_seed(42)

# Path ke direktori dataset
dataset_dir = 'C:/Users/ASUS/Desktop/JST Khodir-20240529T034959Z-001/JST Khodir/dataset'

# Pra-pemrosesan data dan augmentasi
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest',
    rescale=1./255,  # Normalisasi piksel
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split data untuk validasi
)

# Memuat dataset
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),  # Ukuran gambar yang konsisten
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Membangun model CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Mengompilasi model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Melatih model
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Simpan model
model.save('my_model.keras')

# Cetak val_loss, train_loss, val_acc, dan train_acc
val_loss = history.history['val_loss']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
train_acc = history.history['accuracy']

# Membuat plot
epochs = range(1, len(val_loss) + 1)
plt.figure(figsize=(12, 12))
plt.subplot(2, 1, 1)
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.plot(epochs, train_loss, 'r', label='Training Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, val_acc, 'g', label='Validation Accuracy')
plt.plot(epochs, train_acc, 'm', label='Training Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Menghitung F1-score, Precision, dan Recall
validation_generator.reset()
y_true = validation_generator.classes
y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

f1_weighted = f1_score(y_true, y_pred_classes, average='weighted')
precision_weighted = precision_score(y_true, y_pred_classes, average='weighted')
recall_weighted = recall_score(y_true, y_pred_classes, average='weighted')

# Menampilkan hasil F1-score, Precision, dan Recall
plt.figure(figsize=(6, 6))
plt.bar(['F1-score', 'Precision', 'Recall'], [f1_weighted, precision_weighted, recall_weighted])
plt.title('Metrics')
plt.xlabel('Metrics')
plt.ylabel('Score')

# Menampilkan plot
plt.tight_layout()
plt.show()
