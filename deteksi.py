import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Memuat model
model = load_model('my_model.keras')

# Path ke direktori gambar jeruk yang diuji
uji_dir = 'C:\\Users\\ASUS\\Desktop\\JST Khodir-20240529T034959Z-001\JST Khodir\\uji'

# Mendapatkan daftar file gambar dalam folder uji
uji_files = os.listdir(uji_dir)

# Inisialisasi list untuk menyimpan hasil prediksi dan label klasifikasi
predictions = []
labels = []

# Loop melalui setiap file gambar
for uji_file in uji_files:
    # Path lengkap ke gambar
    uji_path = os.path.join(uji_dir, uji_file)

    # Memuat dan memproses gambar
    gambar = load_img(uji_path, target_size=(150, 150))
    gambar_array = img_to_array(gambar)
    gambar_array = np.expand_dims(gambar_array, axis=0)
    gambar_array /= 255.0

    # Melakukan prediksi
    prediksi = model.predict(gambar_array)
    kelas = np.argmax(prediksi, axis=1)[0]

    # Menentukan label klasifikasi
    if kelas == 0:
        label = 'Blackspot'
    elif kelas == 1:
        label = 'Cancer'
    else:
        label = 'Sehat'

    # Menambahkan hasil prediksi dan label ke list
    predictions.append(prediksi[0])
    labels.append(label)

# Menampilkan gambar dan label klasifikasi
plt.figure(figsize=(12, 8))
columns = 3
rows = int(np.ceil(len(uji_files) / columns))

for i, uji_file in enumerate(uji_files):
    # Path lengkap ke gambar
    uji_path = os.path.join(uji_dir, uji_file)

    # Memuat gambar
    gambar = load_img(uji_path, target_size=(150, 150))

    # Menampilkan gambar dengan label klasifikasi
    plt.subplot(rows, columns, i + 1)
    plt.imshow(gambar)
    plt.title(labels[i])
    plt.axis('off')

# Menambahkan teks "Copyright: USN Kolaka"
plt.text(1, -20, "Copyright: Moh. Abdul Khodir",
         fontsize=10, color='gray', ha='right', va='bottom')

plt.tight_layout()
plt.show()
