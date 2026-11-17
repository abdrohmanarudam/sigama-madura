# PANDUAN DEPLOYMENT KE GITHUB & STREAMLIT COMMUNITY CLOUD

**Aplikasi:** SIGAMA - Madura Child Nutrition Screening System (Sistem Informasi Gizi Anak MAdura)  
**Subjudul:** Sistem Informasi Gizi Anak Madura: Skrining Status Gizi Balita Berbasis K-Nearest Neighbors (KNN)  
**Pembuat:** ABD ROHMAN - 20251230035  
**Instansi:** Sistem Informasi, Fakultas Teknik, Universitas Islam Madura 2026  

---

Panduan ini berisi langkah-langkah lengkap dan terperinci untuk mengunggah proyek ke repository **GitHub** serta mempublikasikannya secara online di **Streamlit Community Cloud** (dapat diakses publik secara gratis 24/7).

---

## BAGIAN A: Menghubungkan dan Mengunggah ke GitHub

### 1. Masuk atau Daftar Akun GitHub
1. Buka peramban (browser) dan kunjungi [https://github.com](https://github.com).
2. Masuk (*Sign in*) atau buat akun baru (*Sign up*) menggunakan alamat email resmi: **`abdrohmanuim@gmail.com`**.
3. Pastikan email telah terverifikasi melalui menu **Settings > Emails** di profil GitHub Anda.

### 2. Memasang dan Mengonfigurasi Git di Komputer
1. Pastikan Git telah terpasang di sistem operasi Anda. Jika belum, unduh dari [https://git-scm.com](https://git-scm.com) dan pasang sesuai petunjuk installer.
2. Buka terminal (Linux/macOS) atau Git Bash / Command Prompt (Windows).
3. Atur identitas commit Anda dengan perintah berikut:
   ```bash
   git config --global user.name "Abd Rohman"
   git config --global user.email "abdrohmanuim@gmail.com"
   ```
   > **Penting:** Alamat email `abdrohmanuim@gmail.com` harus sama persis dengan email yang terdaftar di akun GitHub agar seluruh commit tercatat resmi sebagai kontribusi pada profil Anda.

### 3. Membuat Repository Baru di GitHub
1. Di halaman utama GitHub Anda, klik ikon **`+`** di pojok kanan atas, lalu pilih **New repository**.
2. Masukkan informasi repository:
   - **Repository name:** `status-gizi-balita-madura`
   - **Description:** `Smart Nutrition Status Classification System for Madura - SPK Skrining Status Gizi Balita Berbasis KNN`
   - **Visibility:** Pilih **Public** (agar dapat di-deploy di Streamlit Community Cloud gratis).
   - **Initialize this repository with:** **JANGAN centang** opsi *Add a README file*, *Add .gitignore*, maupun *Choose a license*. Biarkan repository benar-benar kosong (*empty repository*).
3. Klik tombol hijau **Create repository**.

### 4. Menghubungkan Folder Proyek Lokal ke GitHub
Buka terminal di dalam folder proyek (`status-gizi-balita-madura`), lalu jalankan urutan perintah berikut:

```bash
# 1. Inisialisasi repository Git lokal (jika belum ada folder .git)
git init

# 2. Tambahkan seluruh berkas ke staging area
git add .

# 3. Buat commit perdana
git commit -m "Initial commit: Smart Nutrition Status Classification System for Madura"

# 4. Pastikan branch utama bernama 'main'
git branch -M main

# 5. Hubungkan ke remote repository GitHub (ganti <USERNAME_GITHUB> dengan username akun Anda)
git remote add origin https://github.com/<USERNAME_GITHUB>/status-gizi-balita-madura.git

# 6. Unggah seluruh berkas ke GitHub
git push -u origin main
```

### 5. Autentikasi Saat Menjalankan Git Push
Ketika menjalankan perintah `git push`, GitHub akan meminta autentikasi:
- **Metode 1 (Git Credential Manager / Web Browser):** Jendela login browser akan terbuka secara otomatis. Masuk ke akun GitHub Anda dan klik *Authorize*.
- **Metode 2 (Personal Access Token - PAT):**
  1. Di GitHub, buka **Settings > Developer settings > Personal access tokens > Tokens (classic)**.
  2. Klik **Generate new token (classic)**, beri nama (misal `deploy-token`), atur masa berlaku, dan centang izin lingkup **`repo`**.
  3. Salin token tersebut dan gunakan sebagai kata sandi (*password*) ketika diminta oleh terminal Git.
- **Metode 3 (GitHub CLI):** Jalankan perintah `gh auth login` di terminal dan ikuti panduan autentikasi di layar.

### 6. Alternatif Menggunakan Antigravity / Visual Studio Code
Bila Anda menggunakan Antigravity IDE atau Visual Studio Code:
1. Buka tab panel **Source Control** di sisi kiri.
2. Klik tombol **Initialize Repository**.
3. Ketik pesan commit: `Initial commit: Smart Nutrition Status Classification System for Madura`, lalu klik centang **Commit**.
4. Klik tombol **Publish Branch** atau **Publish to GitHub**, pilih opsi **Public repository**, lalu lakukan login browser ketika diminta.

### 7. Memverifikasi Berkas di Halaman GitHub
Setelah proses push selesai, buka halaman repository GitHub Anda di browser:
`https://github.com/<USERNAME_GITHUB>/status-gizi-balita-madura`

Pastikan berkas-berkas esensial berikut terlihat lengkap:
- `app.py`
- `train_model.py`
- `check_all.py`
- `styles.py`
- `ui_components.py`
- `requirements.txt`
- `data_status_gizi_balita.xlsx` *(nama berkas harus huruf kecil semua)*
- `models/knn_gizi_balita_madura.joblib`
- `.streamlit/config.toml`
- `README.md`
- `PANDUAN_DEPLOY.md`
- `LAPORAN_PEMERIKSAAN.md`

### 8. Memperbarui Kode di Masa Mendatang
Setiap kali Anda mengubah kode atau memperbarui berkas di lokal, perbarui GitHub dengan:
```bash
git add .
git commit -m "Deskripsi pembaruan fitur atau perbaikan"
git push
```

---

## BAGIAN B: Menghubungkan dan Men-Deploy ke Streamlit Community Cloud

### 1. Masuk ke Streamlit Community Cloud
1. Kunjungi situs resmi: [https://share.streamlit.io](https://share.streamlit.io).
2. Klik tombol **Continue with GitHub**.
3. Masuk menggunakan akun GitHub yang terhubung dengan email `abdrohmanuim@gmail.com`.

### 2. Memberikan Izin Akses (Authorization)
1. Layar otorisasi Streamlit akan muncul menanyakan izin akses ke repository GitHub Anda.
2. Anda dapat memilih memberikan akses ke seluruh repository atau hanya repository `status-gizi-balita-madura`.
3. Klik tombol hijau **Authorize streamlit**.

### 3. Membuat Aplikasi Baru (Create App)
1. Di dashboard Streamlit Cloud, klik tombol **Create app** di pojok kanan atas.
2. Pilih opsi **Deploy a public app from GitHub**.

### 4. Mengisi Formulir Konfigurasi Deployment
Isi formulir aplikasi dengan rincian berikut:
- **Repository:** `<USERNAME_GITHUB>/status-gizi-balita-madura`
- **Branch:** `main`
- **Main file path:** `app.py`
- **App URL (Kustomisasi nama domain):**
  Anda dapat menentukan subdomain gratis, misalnya:
  `status-gizi-balita-madura` sehingga URL aplikasi menjadi:
  `https://status-gizi-balita-madura.streamlit.app`

### 5. Pengaturan Tingkat Lanjut (Advanced Settings)
1. Klik tautan **Advanced settings** sebelum menekan tombol Deploy.
2. Pilih versi Python: **Python 3.11** atau **Python 3.12** (proyek ini sepenuhnya kompatibel).
3. Bagian Secrets dapat dikosongkan karena sistem tidak memerlukan API key eksternal.
4. Klik **Save**.

### 6. Proses Deployment
1. Klik tombol **Deploy!**.
2. Streamlit Cloud akan membaca `requirements.txt`, mengunduh pustaka dependensi (scikit-learn, pandas, numpy, plotly, openpyxl, joblib), dan menjalankan `app.py`.
3. Tunggu 1 hingga 3 menit hingga status berubah dari *Provisioning / In Progress* menjadi aktif.
4. Aplikasi web Anda kini resmi dapat diakses secara langsung melalui link publik tersebut!

### 7. Integrasi CI/CD Otomatis & Reboot
- **Continuous Deployment (CI/CD):** Setiap kali Anda melakukan `git push` ke branch `main`, Streamlit Cloud akan mendeteksi perubahan tersebut secara otomatis dan memperbarui aplikasi web secara langsung tanpa henti (*zero-downtime*).
- **Reboot Aplikasi:** Jika sewaktu-waktu aplikasi perlu dimulai ulang, klik ikon menu tiga titik (**⋮**) di pojok kanan bawah aplikasi di Streamlit Cloud, lalu pilih **Reboot app**.

### 8. Troubleshooting / Masalah Umum & Solusinya
1. **Error: `ModuleNotFoundError` saat build:**
   - *Penyebab:* Ada pustaka yang tertinggal di `requirements.txt`.
   - *Solusi:* Periksa kembali berkas `requirements.txt` dan pastikan mencantumkan semua modul esensial (`streamlit`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `openpyxl`, `plotly`).
2. **Error: File `data_status_gizi_balita.xlsx` tidak ditemukan:**
   - *Penyebab:* Penulisan huruf kapital/kecil tidak cocok atau berkas terabaikan oleh git.
   - *Solusi:* Pastikan nama berkas tepat `data_status_gizi_balita.xlsx` (semua huruf kecil) dan berkas tersebut sudah ter-push ke GitHub (cek melalui browser).
3. **Versi scikit-learn berbeda pada server Streamlit Cloud:**
   - *Penyebab:* Versi scikit-learn di server cloud berbeda dengan komputer lokal saat file `.joblib` disimpan.
   - *Solusi:* Sistem ini telah dilengkapi modul proteksi otomatis: jika `.joblib` gagal dimuat, fungsi `load_active_model` di `app.py` akan secara otomatis melatih ulang model di memori server dan melanjutkan proses tanpa crash!
4. **Data balita baru yang ditambahkan hilang setelah server reboot:**
   - *Penyebab:* Streamlit Cloud menggunakan sistem berkas kontainer sementara (*ephemeral storage*). Jika kontainer dimatikan atau me-reboot, disk kembali ke versi commit terakhir di Git.
   - *Solusi:* Jika ingin menyimpan data balita baru secara permanen ke cloud, unduh berkas dataset melalui tombol unduh di halaman **Manajemen Dataset**, simpan di komputer lokal, lalu lakukan commit dan `git push` ke GitHub.
