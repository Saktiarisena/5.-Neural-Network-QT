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

Sumber Refrensi :

- https://www.kaggle.com/code/satyaprakashshukl/rice-classification
  
[1]	K. H. Ng, S. C. Liew, and F. Ernawan, “An Improved Image Steganography Scheme Based on RDWT and QR Decomposition,” IOP Conf. Ser. Mater. Sci. Eng., vol. 769, no. 1, pp. 222–231, 2020. 

[2]	B. Ando, S. Baglio, S. Castorina, R. Crispino, and V. Marletta, “A Methodology for the Development of Low-Cost, Flexible Touch Sensor for Application in Assistive Technology,” IEEE Trans. Instrum. Meas., vol. 70, 2021. 

[3]	V. Krishnasamy and S. Venkatachalam, “An efficient data flow material model based cloud authentication data security and reduce a cloud storage cost using Index-level Boundary Pattern Convergent Encryption algorithm,” Mater. Today Proc., vol. 81, no. 2, pp. 931–936, 2021. 

[4]	X. Yang et al., “A Survey on Smart Agriculture: Development Modes, Technologies, and Security and Privacy Challenges,” IEEE/CAA J. Autom. Sin., vol. 8, no. 2, pp. 273–302, 2021. 

[5]	S. Ibrahim, S. B. A. Kamaruddin, A. Zabidi, and N. A. M. Ghani, “Contrastive analysis of rice grain classification techniques: Multi-class support vector machine vs artificial neural network,” IAES Int. J. Artif. Intell., vol. 9, no. 4, pp. 616–622, 2020. 

[6]	A. S. Hamzah and A. Mohamed, “Classification of white rice grain quality using ann: A review,” IAES Int. J. Artif. Intell., vol. 9, no. 4, pp. 600–608, 2020. 

[7]	MUH ZAINAL ALTIM, FAISAL, SALMIAH, KASMAN, ANDI YUDHISTIRA, and RITA AMALIA SYAMSU, “Pengklasifikasi Beras Menggunakan Metode Cnn (Convolutional Neural Network),” J. INSTEK (Informatika Sains dan Teknol., vol. 7, no. 1, pp. 151–155, 2022. 

[8]	P. S. Sampaio, A. S. Almeida, and C. M. Brites, “Use of artificial neural network model for rice quality prediction based on grain physical parameters,” Foods, vol. 10, no. 12, 2021. 

[9]	W. Xia, R. Peng, H. Chu, X. Zhu, Z. Yang, and ..., “An Overall Real-Time Mechanism for Classification and Quality Evaluation of Rice,” Available SSRN …. 

[10]	A. Bhattacharjee, K. R. Singh, T. S. Singh, S. Datta, U. R. Singh, and G. Thingbaijam, “INTELLIGENT SYSTEMS AND APPLICATIONS IN ENGINEERING A Comparative Study on Rice Grain Classification Using Convolutional Neural Network and Other Machine Learning Techniques,” pp. 0–1, 2024. 

[11]	T. T. H. Phan, Q. T. Vo, and H. Du Nguyen, “A novel method for identifying rice seed purity using hybrid machine learning algorithms,” Heliyon, vol. 10, no. 14, 2024. 

[12]	Y. Wang, H. Wang, and Z. Peng, “Rice diseases detection and classification using attention based neural network and bayesian optimization,” Expert Syst. Appl., vol. 178, 2021. 

[13]	Y. Haddad, K. Salonitis, and C. Emmanouilidis, “A decision-making framework for the design of local production networks under largescale disruptions,” Procedia Manuf., vol. 55, no. C, pp. 393–400, 2021. 

[14]	I. Samarakoon and P. Liyanage, “Impact of Data Analytics on Operations Success of Apparel Sector ABC Clothing Pvt Limited (Sri Lanka),” Int. J. Comput. Appl., vol. 184, no. 33, pp. 1–15, 2022. 

[15]	Q. W. Kong, J. He, Z. W. Zhang, H. Zheng, and P. Z. Wang, “Projection as a way of thinking to find factors in factor space,” Procedia Comput. Sci., vol. 199, pp. 503–508,

---
