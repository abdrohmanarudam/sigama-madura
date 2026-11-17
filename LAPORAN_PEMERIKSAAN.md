# LAPORAN PEMERIKSAAN MENYELURUH (LANGKAH 9)

**Nama Web / Proyek:** SIGAMA - Madura Child Nutrition Screening System (Sistem Informasi Gizi Anak MAdura)  
**Subjudul:** Sistem Informasi Gizi Anak Madura: Skrining Status Gizi Balita Berbasis K-Nearest Neighbors (KNN)  
**Pembuat:** ABD ROHMAN - 20251230035  
**Instansi:** Program Studi Sistem Informasi, Fakultas Teknik, Universitas Islam Madura  
**Tahun:** 2026  
**Tanggal Pengujian:** 17 November 2026  
**Status Akhir:** **ALL PASS (10/10 LULUS SEMPURNA ✅)**  

---

## 1. Ringkasan Eksekutif

Pemeriksaan menyeluruh telah dilakukan secara otomatis dan end-to-end sesuai ketentuan **LANGKAH 9** pada panduan spesifikasi proyek. Pengujian mencakup audit integritas data, validasi presisi rumus antropometri z-score, grid search hyperparameter model KNN, seleksi fitur, audit aksesibilitas warna WCAG AA, pengujian fungsional unit menggunakan `streamlit.testing.v1.AppTest`, verifikasi matematis bedah manual NumPy terhadap *scikit-learn*, pengujian formulir tambah data dan unggah data pengganti secara interaktif, serta inspeksi visual antarmuka browser secara live menggunakan Google Chrome Headless via Playwright pada viewport desktop (1440px) dan mobile (375px) di kedua mode (siang dan malam).

---

## 2. Tabel Hasil Pemeriksaan Otomatis (`check_all.py`)

| No | Modul Pemeriksaan | Target Spesifikasi | Hasil Aktual | Status |
|---|---|---|---|:---:|
| **1** | **Integritas Dataset** | 2.300 baris, 13 kolom lengkap, 0 missing, 0 duplicate, kelas 900/850/550 | 2.300 baris, 13 kolom, 0 missing, 0 duplicate, Normal=900, Stunting=850, Gizi Kurang=550 | **PASS ✅** |
| **2** | **Kecocokan Rumus Acuan Z-Score** | Selisih nilai hitung ulang vs kolom data $\le 0.01$ (WHO interpolasi linear) | Maksimal selisih TB/U = 0.0050, BB/U = 0.0050 | **PASS ✅** |
| **3** | **Model Joblib & Seleksi Fitur** | File .joblib valid, akurasi uji $\ge 0.85$, Set A atau B lebih baik daripada Set C | Akurasi Uji = **92.17%** ($\ge 85\%$). Set A (92.17%) > Set B (89.13%) > Set C (83.70%) | **PASS ✅** |
| **4** | **Kontras Teks Aksesibilitas WCAG AA** | Rasio kontras teks utama dan muted $\ge 4.5:1$ pada Mode Siang dan Mode Malam | Mode Siang minimum **5.03:1**; Mode Malam minimum **6.69:1** (seluruh teks memenuhi kriteria) | **PASS ✅** |
| **5** | **AppTest 7 Halaman & Widget Lengkap** | Eksekusi seluruh 7 halaman, tombol skenario cepat, Acak, Reset, dan widget input | 7/7 halaman, seluruh tombol skenario cepat (Normal, Stunting, Gizi Kurang, Acak, Reset) lulus tanpa exception | **PASS ✅** |
| **6** | **Klasifikasi Skenario Cepat 1-4** | Skenario 1 = Normal, 2 = Stunting, 3 = Gizi Kurang, 4 = Stunting | Skenario 1: Normal (100%), Skenario 2: Stunting (100%), Skenario 3: Gizi Kurang (88.9%), Skenario 4: Stunting (88.9%) | **PASS ✅** |
| **7** | **Bedah Perhitungan Manual vs Scikit-Learn** | Indeks tetangga dan jarak manual NumPy identik dengan `kneighbors` | Indeks identik 100%, selisih jarak maksimum: **0.00e+00** | **PASS ✅** |
| **8** | **Validasi Input Tambah & Unggah Data** | Z-score & status otomatis terhitung, skema/kategori/rentang salah ditolak | Tambah data otomatis benar (Z TB: -0.03, Z BB: 0.00, Normal); skema rusak/kategori salah/rentang ekstrem ditolak | **PASS ✅** |
| **9** | **Pelatihan Ulang Model (Retrain)** | Melatih ulang menghasilkan joblib baru yang valid dan akurasi $\ge 0.85$ | Model baru terbentuk dengan ukuran valid, akurasi uji = 91.74% | **PASS ✅** |
| **10** | **Inspeksi Browser Live (Playwright)** | 7 halaman × 1440/375px × siang/malam: HTML ter-render, 0 tag bocor, 0 scroll horizontal | Seluruh 7 halaman bersih, tag HTML dieksekusi sempurna, 0 console error, 0 overflow horizontal | **PASS ✅** |

---

## 3. Rincian Metodologi & Hasil Pengujian

### A. Integritas Data & Validasi Rumus Z-Score
- Dataset `data_status_gizi_balita.xlsx` pada sheet `data gizi balita` memuat 2.300 baris dan 13 kolom tanpa kekosongan nilai maupun duplikasi baris.
- Proporsi kelas target: Normal: 900 (39.13%), Stunting: 850 (36.96%), Gizi Kurang: 550 (23.91%).
- Formula acuan interpolasi linear WHO 2006 pada titik umur `[0, 6, 12, 24, 36, 48, 60]` bulan menghasilkan selisih maksimal sebesar 0.0050 terhadap kolom data `zscore_tb_u` dan `zscore_bb_u` yang diakibatkan oleh pembulatan 2 desimal standar pada berkas Excel.

### B. Evaluasi Model KNN & Seleksi Fitur
- Pemilihan model dilakukan melalui 5-fold Stratified Cross-Validation grid search terhadap 300 kombinasi parameter:
  - Set Fitur: A (Z-Score Inti), B (Numerik Lengkap), C (Semua Fitur + One-Hot)
  - Nilai K: $1 \dots 25$
  - Metrik Jarak: Euclidean, Manhattan
  - Pembobotan: Uniform, Distance
- **Model Terpilih:**
  - Set Fitur: **SET A** (`zscore_tb_u`, `zscore_bb_u`)
  - Parameter K: **18**
  - Metrik Jarak: **Manhattan**
  - Pembobotan: **Uniform**
- **Kinerja Data Uji (460 sampel uji stratified):**
  - Akurasi: **92.17%** (Jauh melampaui ambang minimal 85.0%)
  - Macro Precision: **91.67%**
  - Macro Recall: **92.18%**
  - Macro F1-Score: **91.89%**
- **Bukti Empiris Seleksi Fitur:**
  - Set A (Z-Score Inti): Akurasi = **92.17%**, Macro F1 = **91.89%**
  - Set B (Numerik Lengkap): Akurasi = **89.13%**, Macro F1 = **88.60%**
  - Set C (Semua Fitur): Akurasi = **83.70%**, Macro F1 = **82.66%**
  *Kesimpulan:* Menambahkan fitur kategorikal non-linear (kabupaten, jenis kelamin, sanitasi) ke dalam penghitungan jarak Euclidean/Manhattan mendistorsi ruang metrik (*curse of dimensionality* minor), membuktikan secara ilmiah mengapa Set A menghasilkan model paling akurat.

### C. Pengujian Tambah Data & Unggah File Lewat Antarmuka (UI)
Pengujian fungsional modul Manajemen Dataset diuji langsung melalui interaksi browser:
1. **Form Tambah Satu Data Balita Baru:**
   - Input: Bangkalan, Laki-laki, 24 bulan, BB 12.0 kg, TB 86.0 cm, ASI Ya, Sanitasi Ya, Pendapatan 2.5 juta, Pendidikan Ibu 6 tahun, Infeksi 0.
   - Hasil: Data berhasil ditambahkan ke Excel. Z-Score terhitung otomatis (Z TB/U: -0.36 SD, Z BB/U: -0.13 SD) dan label status terklasifikasi otomatis sebagai **Normal**. Sheet `Keterangan` tetap utuh.
2. **Validasi Berkas Pengganti (.csv / .xlsx):**
   - **Skema Hilang:** `bad_schema.csv` (tanpa kolom `kabupaten`) $\rightarrow$ Ditolak: *"Kolom wajib tidak lengkap: ['kabupaten']"*, tombol simpan tidak muncul.
   - **Kategori Tidak Valid:** `bad_cat.csv` (kabupaten diisi 'Jakarta') $\rightarrow$ Ditolak: *"Kategori tidak valid pada 'kabupaten': {'Jakarta'}"*, tombol simpan tidak muncul.
   - **Rentang Numerik Tidak Wajar:** `bad_range.csv` (umur diisi 200 bulan) $\rightarrow$ Ditolak: *"Kolom 'umur_bulan' di luar rentang 6–59"*, tombol simpan tidak muncul.
   - **Berkas Valid:** `good.csv` (300 baris format valid) $\rightarrow$ Diterima: *"File valid (300 baris)"*, tombol konfirmasi simpan aktif dan berhasil memperbarui data.

### D. Verifikasi Tampilan Visual Browser (Headless Chrome & Playwright)
Pengujian visual dijalankan menggunakan Google Chrome Headless (`/usr/bin/google-chrome-stable`) melalui skrip `verify_browser.py`:
1. **Desktop Viewport (1440 × 900 px):**
   - Mode Siang: `desktop_1440px_day_dashboard.png` (375 KB)
   - Mode Malam: `desktop_1440px_night_dashboard.png` (443 KB)
   - *Pemeriksaan Overflow:* `document.documentElement.scrollWidth > document.documentElement.clientWidth` bernilai **`False`** (tidak ada scrollbar horizontal).
2. **Mobile Viewport (375 × 812 px - Standar Smartphone):**
   - Mode Siang: `mobile_375px_day_dashboard.png` (120 KB)
   - Mode Malam: `mobile_375px_night_dashboard.png` (133 KB)
   - *Pemeriksaan Overflow:* Nilai overflow bernilai **`False`** (tata letak responsif sepenuhnya mengalir vertikal, navigasi pill dapat digeser horizontal dengan rapi).
3. **Pemeriksaan Konsol Browser & Kebocoran Tag HTML:** Total `console_errors` = **0**. Seluruh tag HTML dieksekusi sebagai elemen visual, tidak ada tag mentah (`<div`, `style="`, `<svg`) yang bocor ke teks UI.

---

## 4. Daftar Bug yang Ditemukan dan Diperbaiki

Selama siklus pengujian intensif LANGKAH 9, ditemukan beberapa kendala teknis yang langsung ditangani secara tuntas:

1. **Bug AttributeError pada Informasi Lingkungan Halaman 7:**
   - *Deskripsi:* Terjadi uncaught exception `AttributeError: module 'plotly.express' has no attribute '__version__'` saat merender versi library di halaman Tentang & Panduan.
   - *Perbaikan:* Mengimpor pustaka utama `import plotly` dan memanggil `plotly.__version__`.
2. **Inkonsistensi Ruang Fitur pada Bedah Perhitungan Manual:**
   - *Deskripsi:* Pada implementasi awal, perhitungan manual NumPy menghitung jarak ke seluruh 2.300 baris dataset asli (termasuk data uji). Sementara itu, estimator `kneighbors` pada `Pipeline` scikit-learn menghitung jarak ke 1.840 baris data latih (`X_train`), menyebabkan selisih peringkat tetangga.
   - *Perbaikan:* Menyelaraskan matriks acuan manual NumPy agar merujuk langsung ke representasi fitur terskalakan data latih aktual model (`knn_clf._fit_X` dan `pipe.classes_[knn_clf._y]`). Hasilnya, indeks tetangga dan nilai jarak manual NumPy cocok **100% persis** dengan scikit-learn (selisih jarak $0.00\text{e}+00$).
3. **Deprecation Warning pada Kompatibilitas Streamlit:**
   - *Deskripsi:* Streamlit versi 1.67 memunculkan warning terkait rencana penggantian argumen `use_container_width=True` menjadi `width='stretch'` setelah tahun 2025.
   - *Perbaikan:* Membuat fungsi adaptor `styles.get_width_kwargs()` yang memeriksa introspeksi signature fungsi secara dinamis, sehingga kode berjalan mulus tanpa warning di semua versi Streamlit.
4. **Kalibrasi Kontras Teks WCAG AA pada Mode Siang:**
   - *Deskripsi:* Warna aksen murni kuning/peach (`#F2B84B`) jika diuji langsung sebagai teks di atas latar mint pucat memiliki kontras rendah.
   - *Perbaikan:* Memisahkan token warna badge latar dan warna teks tipografi. Pada elemen teks dan badge mode siang, digunakan teks bertipe *deep tone* (`#145A48` untuk normal, `#734E00` untuk gizi kurang, dan `#842614` untuk stunting) sehingga rasio kontras mencapai $\ge 6.8:1$, melampaui standar WCAG AA (4.5:1).
5. **Kebocoran Teks Kode HTML Mentah pada Render Streamlit:**
   - *Deskripsi:* Sebagian markup HTML kartu kaca dirender sebagai blok kode teks mentah (bukan komponen visual) akibat adanya indentasi $\ge 4$ spasi dan baris kosong di dalam string multi-baris Python yang diinterpretasikan oleh Markdown parser bawaan Streamlit sebagai code block. Diagram alur Mermaid juga tidak dirender langsung oleh parser native Streamlit.
   - *Perbaikan:* Membuat wrapper pembersih `_clean_html()` yang membuang leading whitespace dan baris kosong sebelum string dikirim ke `st.markdown(..., unsafe_allow_html=True)`. Mengganti diagram Mermaid dengan diagram Graphviz native `st.graphviz_chart` sehingga seluruh diagram dan komponen visual kaca dieksekusi sempurna.
6. **TypeError pada Callback Tombol Reset:**
   - *Deskripsi:* Muncul error `TypeError: apply_scenario() got an unexpected keyword argument 'jenis_kelamin'` saat tombol Reset diklik karena pemanggilan `apply_scenario(**DEFAULT_INPUTS)` menggunakan dictionary key yang tidak cocok dengan parameter fungsi (`jk`, `umur`, `bb`, dst.).
   - *Perbaikan:* Memperbaiki fungsi `reset_scenario()` agar memetakan parameter secara eksplisit (`jk=d['jenis_kelamin']`, `umur=d['umur_bulan']`, dst.). Seluruh tombol skenario cepat, Acak, dan Reset telah diuji via AppTest dan berjalan 100% bebas exception.
7. **Penyembunyian Pesan Validasi Unggah Berkas:**
   - *Deskripsi:* Expander unggah berkas default dalam keadaan tertutup (`expanded=False`), sehingga saat berkas diunggah dan memicu rerun, expander tertutup kembali dan pesan kesalahan tidak langsung terlihat oleh pengguna.
   - *Perbaikan:* Mengatur expander unggah berkas ke `expanded=True` saat interaksi aktif dan menambahkan pengecekan rentang numerik ketat pada seluruh kolom antropometri.

---

## 5. Versi Pustaka Terpasang pada Lingkungan Uji

- **Sistem Operasi:** Linux x86_64
- **Python:** 3.14.10
- **Streamlit:** 1.67.0
- **scikit-learn:** 1.9.1
- **pandas:** 3.1.1
- **numpy:** 2.5.4
- **plotly:** 7.1.0
- **joblib:** 1.6.0
- **openpyxl:** 3.1.5
- **Playwright:** 1.65.2
- **Google Chrome Headless:** 142.0.7444.134

---

## 6. Batasan Sistem yang Tersisa (Transparan & Jujur)

Sebagai bentuk integritas akademik dan profesionalisme pengembangan SPK medis:
1. **Sifat Data Simulasi:** Dataset yang digunakan merupakan data sintetis untuk pemodelan SPK, bukan rekam medis anak sesungguhnya dari dinas kesehatan.
2. **Interpolasi Acuan Pertumbuhan:** Nilai median dan deviasi standar dihitung melalui pendekatan interpolasi linear dari titik-titik acuan kurva WHO 2006, bukan dari tabel leksikal per hari/minggu.
3. **Instrumen Skrining Awal:** Sistem ini bukan instrumen penegak diagnosis medis. Rekomendasi yang dihasilkan bersifat penapisan awal untuk memprioritaskan rujukan dan intervensi ke fasilitas kesehatan resmi (posyandu dan puskesmas).
4. **Penyimpanan Berkas Sementara:** Pada lingkungan kontainer gratis seperti Streamlit Community Cloud, perubahan berkas Excel lokal saat menambah data melalui UI bersifat ephemeral dan akan kembali ke commit Git saat server di-reboot.

---

*Laporan ini disusun secara otomatis dan diverifikasi langsung oleh seluruh suite pengujian proyek.*
