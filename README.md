## ✨ Fitur

* Memuat dan membaca dataset beras (format `.csv`)
* Melatih jaringan syaraf tiruan satu lapis (tanpa library eksternal ML)
* Pengaturan learning rate dan jumlah epoch
* Visualisasi perbandingan antara prediksi dan label sebenarnya
* Menampilkan akurasi dan mean squared error (MSE)
* Menampilkan daftar nama kelas

---

## 📊 Format Dataset

Pastikan dataset Anda memiliki **header** dan 5 kolom dengan urutan sebagai berikut:

```csv
Solidity,AspectRatio,Roundness,Compactness,Class
0.90,1.52,0.85,0.87,Jasmine
...
```

Lokasi file default: `data/Rice_MSC_Dataset_sample.csv`

---

## 🚀 Cara Menjalankan

### 1. Clone repositori ini

```bash
git clone https://github.com/usernameanda/rice-classifier.git
cd rice-classifier
```

### 2. Siapkan dataset

Letakkan file CSV Anda di folder `data/`. Anda dapat mengganti nama file atau mengubah path-nya di fungsi `RiceApp::default()`.

### 3. Build dan jalankan

Pastikan Anda telah menginstal Rust: [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install)

```bash
cargo run --release
```

---

## 🧠 Tentang Jaringan Syaraf Tiruan

Model ini adalah neural network satu lapis dengan aktivasi softmax dan pembaruan bobot menggunakan gradient descent manual:

* Input: 4 fitur (Solidity, Aspect Ratio, Roundness, Compactness)
* Output: Probabilitas kelas (satu-hot encoding)
* Loss: Menggunakan turunan dari softmax dan error
* Semua logika diimplementasikan murni dengan Rust tanpa library ML eksternal

---

## 📦 Dependensi

* `eframe` / `egui` – Framework GUI
* `ndarray` – Operasi array/matriks
* `csv` – Pembaca file CSV
* `linfa` – Untuk membagi dataset
* `rand` – Inisialisasi bobot acak

---

## 📈 Contoh Output (Konsol)

```
Classification Results:
Index      Prediksi        Aktual          Benar      
0          Jasmine         Jasmine         ✓         
1          Basmati         Guntur          ✗         
...

Model Performance:
Akurasi: 84.50%
Prediksi benar: 169/200
Mean Squared Error: 0.3275

Class Mapping:
Class 0: Basmati
Class 1: Guntur
Class 2: Jasmine
```

---

## 📂 Struktur Proyek

```
src/
├── main.rs             # Titik masuk program dan logika GUI
data/
└── Rice_MSC_Dataset_sample.csv
```

---
