# SIGAMA - Madura Child Nutrition Screening System
### (Sistem Informasi Gizi Anak MAdura)

> **Sistem Informasi Gizi Anak Madura: Skrining Status Gizi Balita Berbasis K-Nearest Neighbors (KNN)**

---

### 👨‍💻 Identitas Pembuat
- **Nama Pengembang:** ABD ROHMAN
- **NIM:** 20251230035
- **Program Studi:** Sistem Informasi
- **Fakultas:** Fakultas Teknik
- **Instansi:** Universitas Islam Madura
- **Tahun:** 2026

---

## 📌 1. Deskripsi Proyek

**SIGAMA (Sistem Informasi Gizi Anak MAdura) - Madura Child Nutrition Screening System** adalah aplikasi web Sistem Pendukung Keputusan (SPK) yang dirancang untuk membantu penapisan (*skrining*) status gizi balita di 4 kabupaten Pulau Madura (Bangkalan, Sampang, Pamekasan, dan Sumenep). 

Aplikasi mengimplementasikan algoritma pembelajaran mesin **K-Nearest Neighbors (KNN)** melalui pustaka `scikit-learn` dengan antarmuka modern bergaya **Glassmorphism Lembut & Tiga Dimensi**, mendukung tema **Cerah Adem** dengan mode Siang & Malam, jam digital *real-time*, visualisasi interaktif 3D & 2D (Plotly), serta modul **Bedah Perhitungan Manual** yang transparan dan dapat diverifikasi 100%.

> ⚠️ **Catatan Etika & Batasan Sistem:**  
> Sistem ini merupakan alat bantu **skrining awal** berbasis data simulasi (sintetis), **BUKAN alat diagnosis medis**. Hasil penapisan bersifat perkiraan. Konfirmasi pengukuran antropometri dan intervensi penanganan wajib dilakukan oleh tenaga kesehatan resmi di posyandu atau puskesmas.

---

## ✨ 2. Fitur Utama Aplikasi

1. **🧒 Dashboard & Skrining Gizi:**
   - Kartu status klasifikasi utama (*Normal*, *Stunting*, *Gizi Kurang*) dengan tingkat keyakinan (*confidence level*) dari `predict_proba`.
   - Pengukur z-score visual multi-zona horizontal berskala $-4 \dots +3\text{ SD}$.
   - Evaluasi faktor determinan sosial & infeksi (ASI eksklusif, sanitasi layak, infeksi, pendapatan keluarga).
   - Riwayat skrining sesi pengguna (hingga 50 data) dengan fitur unduh CSV.
2. **📊 Visualisasi Ruang Fitur 3D & 2D:**
   - Scatter plot 3 dimensi interaktif (Plotly) dengan penanda posisi balita aktif secara khusus.
   - Scatter plot 2D Z-Score TB/U vs BB/U dilengkapi garis batas ambang $-2\text{ SD}$.
   - Radar diagram profil balita terhadap rata-rata tiap kelas gizi.
   - Distribusi status gizi per kabupaten di Madura dan faktor risiko.
3. **🧮 Bedah Perhitungan Manual (Transparan):**
   - Langkah 1: Perhitungan Z-score dengan formula $\LaTeX$ dari interpolasi linear kurva WHO 2006.
   - Langkah 2: Standardisasi fitur (`StandardScaler`) $\mu$ dan $\sigma$ dengan angka riil.
   - Langkah 3: Perhitungan jarak manual (Manhattan / Euclidean) menggunakan vektorisasi NumPy ke data latih.
   - Langkah 4: Pemungutan suara (*voting*) tetangga terdekat.
   - Langkah 5: Pembuktian kecocokan matematis dengan fungsi `kneighbors` scikit-learn (identik 100%).
4. **📈 Evaluasi & Optimasi K:**
   - Tuning hyperparameter interaktif: rasio data uji (10–40%), rentang K (1–30), metrik jarak, pembobotan, dan Stratified CV folds.
   - Grafik K vs Akurasi, Confusion Matrix, Classification Report, serta tabel perbandingan Set Fitur A/B/C.
   - Tombol simpan konfigurasi terbaik ke file `.joblib` dan tombol unduh model.
5. **📁 Manajemen Dataset:**
   - Tabel data kaca responsif dengan pencarian kata kunci, filter kabupaten/gender/status, dan paginasi (50 data/halaman).
   - Formulir tambah balita baru dengan penghitungan z-score dan status gizi otomatis saat disimpan ke Excel.
   - Validasi skema ketat berkas unggahan (.xlsx / .csv).
   - Fitur unduh dataset (CSV & Excel) dan tombol pemulihan dataset asli.
6. **🥗 Edukasi Gizi & Kurva Pertumbuhan:**
   - Kurva pertumbuhan interaktif TB/U dan BB/U standar WHO 2006 dengan titik balita aktif.
   - Edukasi 1000 Hari Pertama Kehidupan (HPK), ASI eksklusif, protein hewani, dan sanitasi.
   - Panduan gizi berbasis potensi pangan lokal Madura (ikan laut segar, telur, daun kelor).
7. **ℹ️ Tentang & Panduan:**
   - Profil lengkap pengembang, flowchart arsitektur algoritma, panduan penggunaan langkah demi langkah, glosarium, dan informasi versi pustaka.

---

## 🔬 3. Pemodelan Machine Learning & Seleksi Fitur

Hasil pengujian komprehensif membuktikan bahwa seleksi fitur sangat berpengaruh terhadap performa algoritma KNN:
- **SET A (Z-Score Inti):** `zscore_tb_u`, `zscore_bb_u` -> Akurasi Uji = **92.17%**, Macro F1 = **91.89%**
- **SET B (Numerik Lengkap):** Fitur numerik antropometri, ekonomi, dan infeksi -> Akurasi Uji = **89.13%**, Macro F1 = **88.60%**
- **SET C (Semua Fitur):** Termasuk one-hot kategorikal (kabupaten, gender, ASI, sanitasi) -> Akurasi Uji = **83.70%**, Macro F1 = **82.66%**

> **Konfigurasi Model Aktif Terbaik:**
> - Set Fitur: **SET A**
> - Parameter K: **18**
> - Metrik Jarak: **Manhattan**
> - Pembobotan: **Uniform**
> - Akurasi Data Uji: **92.17%** (Target: $\ge 85\%$)

---

## 📂 4. Struktur Berkas Proyek

```
status-gizi-balita-madura/
├── app.py                               # Aplikasi utama Streamlit (7 halaman navigasi)
├── train_model.py                        # Skrip pelatihan model KNN & evaluasi Grid Search
├── check_all.py                          # Skrip otomatis audit & verifikasi menyeluruh (9/9 PASS)
├── verify_browser.py                     # Skrip otomatis uji visual Chrome Headless & Playwright
├── styles.py                             # Gaya Glassmorphism CSS, tema Siang/Malam, jam real-time
├── ui_components.py                      # Komponen tabel kaca HTML, pengukur Z-score, kartu hasil
├── requirements.txt                      # Daftar pustaka dependensi Python
├── README.md                             # Dokumentasi utama proyek
├── PANDUAN_DEPLOY.md                     # Panduan komprehensif deployment GitHub & Streamlit Cloud
├── LAPORAN_PEMERIKSAAN.md                # Laporan detail hasil audit dan perbaikan bug
├── .gitignore                            # Konfigurasi pengecualian Git
├── .streamlit/
│   └── config.toml                       # Konfigurasi headless server & tema Streamlit
├── data_status_gizi_balita.xlsx          # Dataset antropometri balita Madura (2.300 sampel)
├── models/
│   └── knn_gizi_balita_madura.joblib     # Model KNN hasil pelatihan terverifikasi
└── screenshots/                          # Tangkapan layar browser live (1440px & 375px)
```

---

## 🚀 5. Cara Menjalankan Aplikasi Secara Lokal

### Prasyarat:
- Python versi 3.10, 3.11, atau 3.12 (kompatibel hingga 3.14).
- PIP (*Python Package Installer*).

### Langkah-langkah:
1. **Clone atau buka folder proyek:**
   ```bash
   cd "status-gizi-balita-madura"
   ```

2. **Buat dan aktifkan virtual environment (opsional namun disarankan):**
   ```bash
   python -m venv venv
   # Linux/macOS:
   source venv/bin/activate
   # Windows:
   venv\Scripts\activate
   ```

3. **Pasang seluruh dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Latih model KNN (jika file .joblib belum tersedia):**
   ```bash
   python train_model.py
   ```

5. **Jalankan aplikasi Streamlit:**
   ```bash
   streamlit run app.py
   ```
   Buka peramban di alamat: `http://localhost:8501`.

6. **Jalankan pengujian otomatis:**
   ```bash
   python check_all.py
   ```

---

## 🌐 6. Panduan Deployment

Panduan terperinci langkah demi langkah untuk menghubungkan akun GitHub (`abdrohmanuim@gmail.com`) dan mempublikasikan aplikasi ke **Streamlit Community Cloud** tersedia pada berkas terpisah:
👉 **[PANDUAN_DEPLOY.md](file:///run/media/eka/New%20Volume/Kuliah/SEMESTER%20III/ROHMAN/TUGAS%204/PANDUAN_DEPLOY.md)**

Laporan audit teknis dan hasil pemeriksaan otomatis 9/9 lulus dapat dilihat pada:
👉 **[LAPORAN_PEMERIKSAAN.md](file:///run/media/eka/New%20Volume/Kuliah/SEMESTER%20III/ROHMAN/TUGAS%204/LAPORAN_PEMERIKSAAN.md)**

---

## 📜 7. Lisensi & Hak Cipta

© 2026 **ABD ROHMAN** (NIM: 20251230035)  
Program Studi Sistem Informasi, Fakultas Teknik, Universitas Islam Madura.  
Seluruh hak cipta dilindungi undang-undang.
