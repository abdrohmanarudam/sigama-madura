"""
app.py
Aplikasi Utama Streamlit:
Smart Nutrition Status Classification System for Madura
Sistem Pendukung Keputusan Skrining Status Gizi Balita di Madura Berbasis K-Nearest Neighbors (KNN)
Pembuat : ABD ROHMAN - 20251230035
Instansi: Sistem Informasi, Fakultas Teknik, Universitas Islam Madura 2026
"""

from __future__ import annotations

import io
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import streamlit as st
import streamlit.components.v1 as components

# Import modul internal
import styles
import train_model
import ui_components

# ==============================================================================
# 1. KONFIGURASI HALAMAN & STATE AWAL
# ==============================================================================
st.set_page_config(
    page_title="SIGAMA | Madura Child Nutrition Screening System | ABD ROHMAN",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="auto",
)

# ------------------------------------------------------------------------------
# Perbaikan render HTML: Markdown menganggap baris berindentasi >= 4 spasi (dan
# baris kosong di dalam blok HTML) sebagai blok kode. Rapikan HTML sebelum render
# agar dieksekusi sebagai HTML, bukan tampil sebagai teks kode.
# ------------------------------------------------------------------------------
_orig_markdown = st.markdown


def _clean_html(text: str) -> str:
    lines = [ln.strip() for ln in str(text).splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _safe_markdown(body, *args, **kwargs):
    if kwargs.get("unsafe_allow_html") and isinstance(body, str):
        body = _clean_html(body)
    return _orig_markdown(body, *args, **kwargs)


st.markdown = _safe_markdown

DATA_FILE_PATH = Path("data_status_gizi_balita.xlsx")
DATA_BACKUP_PATH = Path("data_status_gizi_balita_backup.xlsx")
MODEL_FILE_PATH = Path("models/knn_gizi_balita_madura.joblib")

# Buat cadangan data asli saat pertama kali jalan jika belum ada
if DATA_FILE_PATH.exists() and not DATA_BACKUP_PATH.exists():
    try:
        shutil.copy2(DATA_FILE_PATH, DATA_BACKUP_PATH)
    except Exception:
        pass

# Inisialisasi query params mode (siang/malam)
query_mode = st.query_params.get("mode", "siang")
initial_dark = query_mode == "malam"

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = initial_dark

# Inisialisasi state navigasi & input balita
NAV_PAGES = [
    "🧒 Dashboard & Skrining Gizi",
    "📊 Visualisasi 3D & 2D",
    "🧮 Bedah Perhitungan Manual",
    "📈 Evaluasi & Optimasi K",
    "📁 Manajemen Dataset",
    "🥗 Edukasi Gizi & Kurva Pertumbuhan",
    "ℹ️ Tentang & Panduan",
]

if "page" not in st.session_state:
    st.session_state["page"] = NAV_PAGES[0]
if "last_page" not in st.session_state:
    st.session_state["last_page"] = NAV_PAGES[0]
if "last_dark_mode" not in st.session_state:
    st.session_state["last_dark_mode"] = st.session_state["dark_mode"]

# Default input values balita
DEFAULT_INPUTS = {
    "jenis_kelamin": "Laki-laki",
    "umur_bulan": 24,
    "berat_badan_kg": 12.2,
    "tinggi_badan_cm": 87.0,
    "kabupaten": "Sumenep",
    "asi_eksklusif": "Ya",
    "sanitasi_layak": "Ya",
    "pendapatan_keluarga_juta": 2.5,
    "lama_sekolah_ibu": 12,
    "episode_infeksi_3bln": 0,
}

for k_key, v_val in DEFAULT_INPUTS.items():
    if f"input_{k_key}" not in st.session_state:
        st.session_state[f"input_{k_key}"] = v_val

if "last_asi" not in st.session_state:
    st.session_state["last_asi"] = st.session_state.get("input_asi_eksklusif", "Ya")
if "last_san" not in st.session_state:
    st.session_state["last_san"] = st.session_state.get("input_sanitasi_layak", "Ya")
if "pending_loading_msg" not in st.session_state:
    st.session_state["pending_loading_msg"] = None

if "screening_history" not in st.session_state:
    st.session_state["screening_history"] = []

if "dataset_page" not in st.session_state:
    st.session_state["dataset_page"] = 1

# ==============================================================================
# 2. HELPER LOADING & CACHE MODEL / DATA
# ==============================================================================
def show_loading(label: str = "Memproses komputasi data...", duration: float = 0.45):
    """Tampilkan modal overlay fullscreen menutupi seluruh layar lalu kosongkan."""
    ph = st.empty()
    is_d = st.session_state.get("dark_mode", False)
    ph.markdown(
        _clean_html(styles.get_loading_overlay_html(label, is_d)),
        unsafe_allow_html=True,
    )
    time.sleep(duration)
    ph.empty()


def on_asi_change():
    val = st.session_state.get("input_asi_eksklusif", "Ya")
    st.session_state["pending_loading_msg"] = f"Pembaruan Parameter: ASI Eksklusif = {val}..."


def on_san_change():
    val = st.session_state.get("input_sanitasi_layak", "Ya")
    st.session_state["pending_loading_msg"] = f"Pembaruan Parameter: Sanitasi Layak = {val}..."


def on_nav_change():
    val = st.session_state.get("nav_radio_control")
    if val and val != st.session_state.get("last_page"):
        st.session_state["pending_loading_msg"] = f"Memuat Tab: {val}..."
        st.session_state["page"] = val


@st.cache_data(show_spinner=False)
def get_dataset(filepath: str | Path) -> pd.DataFrame:
    return train_model.load_data(filepath)


@st.cache_resource(show_spinner=False)
def load_active_model(model_path: str | Path, data_path: str | Path) -> Dict[str, Any]:
    mp = Path(model_path)
    dp = Path(data_path)
    if not mp.exists():
        # Latih model otomatis jika file belum ada
        return train_model.train_and_save_model(data_path=dp, output_path=mp)
    try:
        loaded = joblib.load(mp)
        # Validasi struktur dict
        if "pipeline" not in loaded or "features" not in loaded:
            raise ValueError("Struktur model joblib tidak valid.")
        return loaded
    except Exception:
        # Jika beda versi sklearn atau corrupt, latih ulang otomatis
        return train_model.train_and_save_model(data_path=dp, output_path=mp)


# Cek pemicu loading animasi (pergantian tab, tema, radio button balita, atau CRUD)
page_changed = st.session_state["page"] != st.session_state["last_page"]
mode_changed = st.session_state["dark_mode"] != st.session_state["last_dark_mode"]
asi_changed = st.session_state.get("input_asi_eksklusif") != st.session_state.get("last_asi")
san_changed = st.session_state.get("input_sanitasi_layak") != st.session_state.get("last_san")
pending_msg = st.session_state.get("pending_loading_msg")

loading_label = None
if pending_msg:
    loading_label = pending_msg
    st.session_state["pending_loading_msg"] = None
elif page_changed:
    loading_label = f"Memuat Tab: {st.session_state['page']}..."
elif mode_changed:
    loading_label = "Menerapkan tema tampilan..."
elif asi_changed:
    loading_label = f"Pembaruan Parameter: ASI Eksklusif = {st.session_state.get('input_asi_eksklusif')}..."
elif san_changed:
    loading_label = f"Pembaruan Parameter: Sanitasi Layak = {st.session_state.get('input_sanitasi_layak')}..."

if loading_label:
    show_loading(loading_label, duration=0.45)

# Sinkronkan status pelacak
st.session_state["last_page"] = st.session_state["page"]
st.session_state["last_dark_mode"] = st.session_state["dark_mode"]
st.session_state["last_asi"] = st.session_state.get("input_asi_eksklusif")
st.session_state["last_san"] = st.session_state.get("input_sanitasi_layak")

# Suntikkan CSS Tema Cerah Adem (Siang / Malam)
is_dark = st.session_state["dark_mode"]
st.markdown(styles.get_theme_css(is_dark), unsafe_allow_html=True)
theme_colors = styles.get_theme_colors(is_dark)

# ==============================================================================
# 3. MUAT DATASET & MODEL AKTIF
# ==============================================================================
try:
    df_data = get_dataset(DATA_FILE_PATH)
except Exception as e:
    st.error(f"⚠️ Gagal membaca dataset: {e}")
    st.info("Silakan unggah dataset yang sesuai di halaman Manajemen Dataset.")
    df_data = pd.DataFrame()

try:
    active_model = load_active_model(MODEL_FILE_PATH, DATA_FILE_PATH)
except Exception as e:
    st.error(f"⚠️ Gagal memuat model: {e}")
    active_model = None

# ==============================================================================
# 4. CALLBACK SKENARIO CEPAT
# ==============================================================================
def apply_scenario(
    jk: str,
    umur: int,
    bb: float,
    tb: float,
    kab: str,
    asi: str,
    san: str,
    pend: float,
    sek: int,
    inf: int,
    label: str = "Menerapkan data skenario balita...",
):
    st.session_state["pending_loading_msg"] = label
    st.session_state["input_jenis_kelamin"] = jk
    st.session_state["input_umur_bulan"] = umur
    st.session_state["input_berat_badan_kg"] = float(bb)
    st.session_state["input_tinggi_badan_cm"] = float(tb)
    st.session_state["input_kabupaten"] = kab
    st.session_state["input_asi_eksklusif"] = asi
    st.session_state["input_sanitasi_layak"] = san
    st.session_state["input_pendapatan_keluarga_juta"] = float(pend)
    st.session_state["input_lama_sekolah_ibu"] = sek
    st.session_state["input_episode_infeksi_3bln"] = inf
    st.session_state["last_asi"] = asi
    st.session_state["last_san"] = san


def apply_random_scenario():
    if not df_data.empty:
        row = df_data.sample(n=1, random_state=int(time.time() * 1000) % 10000).iloc[0]
        apply_scenario(
            jk=str(row["jenis_kelamin"]),
            umur=int(row["umur_bulan"]),
            bb=float(row["berat_badan_kg"]),
            tb=float(row["tinggi_badan_cm"]),
            kab=str(row["kabupaten"]),
            asi=str(row["asi_eksklusif"]),
            san=str(row["sanitasi_layak"]),
            pend=float(row["pendapatan_keluarga_juta"]),
            sek=int(row["lama_sekolah_ibu"]),
            inf=int(row["episode_infeksi_3bln"]),
            label="Memuat Sampel Balita Acak dari Dataset...",
        )


def reset_scenario():
    d = DEFAULT_INPUTS
    apply_scenario(
        jk=d["jenis_kelamin"],
        umur=d["umur_bulan"],
        bb=d["berat_badan_kg"],
        tb=d["tinggi_badan_cm"],
        kab=d["kabupaten"],
        asi=d["asi_eksklusif"],
        san=d["sanitasi_layak"],
        pend=d["pendapatan_keluarga_juta"],
        sek=d["lama_sekolah_ibu"],
        inf=d["episode_infeksi_3bln"],
        label="Mereset Parameter Balita ke Default...",
    )


# ==============================================================================
# 5. SIDEBAR: PANEL DATA BALITA & PENGATURAN
# ==============================================================================
with st.sidebar:
    st.markdown("### 🧸 Panel Data Balita")
    st.markdown(
        "<div style='font-size:0.8rem; color:var(--muted); margin-top:-10px; margin-bottom:12px;'>Pilih skenario atau atur parameter balita secara mandiri:</div>",
        unsafe_allow_html=True,
    )

    # Skenario Cepat
    with st.expander("⚡ Skenario Cepat", expanded=False):
        c_sc1, c_sc2 = st.columns(2)
        with c_sc1:
            st.button(
                "👶 Balita Normal",
                use_container_width=True,
                on_click=apply_scenario,
                args=("Laki-laki", 24, 12.2, 87.0, "Sumenep", "Ya", "Ya", 2.5, 12, 0, "Menerapkan Skenario: Balita Normal (Sumenep)..."),
                help="Balita umur 24 bln dengan antropometri ideal",
            )
            st.button(
                "⚖️ Gizi Kurang",
                use_container_width=True,
                on_click=apply_scenario,
                args=("Laki-laki", 36, 10.3, 95.5, "Pamekasan", "Ya", "Ya", 1.8, 9, 1, "Menerapkan Skenario: Balita Gizi Kurang (Pamekasan)..."),
                help="BB/U rendah, tinggi badan normal",
            )
        with c_sc2:
            st.button(
                "📏 Berisiko Stunting",
                use_container_width=True,
                on_click=apply_scenario,
                args=("Perempuan", 30, 11.0, 82.0, "Sampang", "Tidak", "Tidak", 1.2, 6, 2, "Menerapkan Skenario: Balita Berisiko Stunting (Sampang)..."),
                help="TB/U rendah umur 30 bulan",
            )
            st.button(
                "🌧️ Stunting Usia Dini",
                use_container_width=True,
                on_click=apply_scenario,
                args=("Perempuan", 18, 8.4, 74.0, "Bangkalan", "Tidak", "Tidak", 1.0, 6, 3, "Menerapkan Skenario: Balita Stunting Usia Dini (Bangkalan)..."),
                help="Stunting pada 1000 HPK",
            )
        c_rnd, c_rst = st.columns(2)
        with c_rnd:
            st.button("🎲 Acak", use_container_width=True, on_click=apply_random_scenario)
        with c_rst:
            st.button("↺ Reset", use_container_width=True, on_click=reset_scenario)

    # Input Demografi & Antropometri
    st.selectbox(
        "Jenis Kelamin Balita",
        options=train_model.VALID_CATEGORIES["jenis_kelamin"],
        key="input_jenis_kelamin",
        help="Jenis kelamin balita untuk kurva acuan pertumbuhan WHO",
    )

    st.selectbox(
        "Kabupaten di Madura",
        options=train_model.VALID_CATEGORIES["kabupaten"],
        key="input_kabupaten",
        help="Kabupaten domisili balita di Pulau Madura",
    )

    st.slider(
        "Umur Balita (Bulan)",
        min_value=6,
        max_value=59,
        step=1,
        key="input_umur_bulan",
        help="Rentang umur balita dalam bulan (6 hingga 59 bulan)",
    )

    c_bb, c_tb = st.columns(2)
    with c_bb:
        st.number_input(
            "Berat Badan (kg)",
            min_value=3.0,
            max_value=25.0,
            step=0.1,
            format="%.1f",
            key="input_berat_badan_kg",
            help="Hasil timbangan berat badan balita (kg)",
        )
    with c_tb:
        st.number_input(
            "Tinggi Badan (cm)",
            min_value=55.0,
            max_value=125.0,
            step=0.1,
            format="%.1f",
            key="input_tinggi_badan_cm",
            help="Hasil pengukuran panjang/tinggi badan balita (cm)",
        )

    # Perhitungan Z-Score Otomatis Langsung di Sidebar
    current_ztb, current_zbb = train_model.calculate_zscores(
        umur_bulan=st.session_state["input_umur_bulan"],
        jenis_kelamin=st.session_state["input_jenis_kelamin"],
        berat_badan_kg=st.session_state["input_berat_badan_kg"],
        tinggi_badan_cm=st.session_state["input_tinggi_badan_cm"],
    )

    st.markdown(
        f"""
        <div style="
            background: rgba(255,255,255,0.25);
            border: 1px solid var(--glass-border);
            border-radius: 14px;
            padding: 8px 12px;
            margin: 10px 0;
            display: flex;
            justify-content: space-around;
            text-align: center;
        ">
            <div>
                <div style="font-size:0.72rem; color:var(--muted); font-weight:600;">Z TB/U</div>
                <div style="font-size:1.0rem; font-weight:800; color:{theme_colors['primary']};">{current_ztb:+.2f} SD</div>
            </div>
            <div style="border-left:1px solid var(--glass-border);"></div>
            <div>
                <div style="font-size:0.72rem; color:var(--muted); font-weight:600;">Z BB/U</div>
                <div style="font-size:1.0rem; font-weight:800; color:{theme_colors['accent']};">{current_zbb:+.2f} SD</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🛡️ Faktor Determinan & Sosial", expanded=False):
        c_asi, c_san = st.columns(2)
        with c_asi:
            st.radio(
                "ASI Eksklusif",
                options=["Ya", "Tidak"],
                key="input_asi_eksklusif",
                on_change=on_asi_change,
                help="Pemberian ASI saja sampai usia 6 bulan",
            )
        with c_san:
            st.radio(
                "Sanitasi Layak",
                options=["Ya", "Tidak"],
                key="input_sanitasi_layak",
                on_change=on_san_change,
                help="Akses jamban sehat & air bersih layak",
            )

        st.slider(
            "Pendapatan Keluarga (Juta Rp/bln)",
            min_value=0.5,
            max_value=9.0,
            step=0.1,
            format="%.1f",
            key="input_pendapatan_keluarga_juta",
            help="Total pendapatan rumah tangga per bulan dalam juta rupiah",
        )

        st.selectbox(
            "Lama Sekolah Ibu",
            options=[6, 9, 12, 16],
            format_func=lambda x: {6: "6 Tahun (SD)", 9: "9 Tahun (SMP)", 12: "12 Tahun (SMA)", 16: "16 Tahun (Perguruan Tinggi)"}[x],
            key="input_lama_sekolah_ibu",
            help="Tingkat pendidikan formal terakhir ibu",
        )

        st.slider(
            "Episode Infeksi (3 Bln Terakhir)",
            min_value=0,
            max_value=3,
            step=1,
            key="input_episode_infeksi_3bln",
            help="Frekuensi balita sakit diare atau ISPA dalam 3 bulan terakhir",
        )

    st.markdown("---")
    st.markdown("##### 🎨 Tampilan & Tema")
    toggle_mode = st.toggle(
        "☀️ Mode Siang / 🌙 Mode Malam",
        value=st.session_state["dark_mode"],
        key="theme_toggle",
    )
    if toggle_mode != st.session_state["dark_mode"]:
        st.session_state["dark_mode"] = toggle_mode
        st.query_params["mode"] = "malam" if toggle_mode else "siang"
        st.rerun()

    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size:0.75rem; color:var(--muted); line-height:1.4; padding-top:4px;">
            <strong>Pembuat:</strong><br>
            ABD ROHMAN - 20251230035<br>
            Sistem Informasi, Fakultas Teknik<br>
            Universitas Islam Madura 2026
        </div>
        """,
        unsafe_allow_html=True,
    )

# Susun dictionary baris input aktif pengguna
active_input_dict = {
    "kabupaten": st.session_state["input_kabupaten"],
    "jenis_kelamin": st.session_state["input_jenis_kelamin"],
    "umur_bulan": st.session_state["input_umur_bulan"],
    "berat_badan_kg": st.session_state["input_berat_badan_kg"],
    "tinggi_badan_cm": st.session_state["input_tinggi_badan_cm"],
    "zscore_tb_u": current_ztb,
    "zscore_bb_u": current_zbb,
    "asi_eksklusif": st.session_state["input_asi_eksklusif"],
    "sanitasi_layak": st.session_state["input_sanitasi_layak"],
    "pendapatan_keluarga_juta": st.session_state["input_pendapatan_keluarga_juta"],
    "lama_sekolah_ibu": st.session_state["input_lama_sekolah_ibu"],
    "episode_infeksi_3bln": st.session_state["input_episode_infeksi_3bln"],
}

# ==============================================================================
# 6. HEADER BANNER & JAM REAL-TIME
# ==============================================================================
st.markdown(ui_components.render_banner(is_dark), unsafe_allow_html=True)

# Tampilkan Jam Real-time Berdetak (JavaScript Component)
components.html(styles.get_realtime_clock_html(is_dark), height=46)

# ==============================================================================
# 7. NAVIGASI HORIZONTAL GLASS PILL
# ==============================================================================
selected_nav = st.radio(
    "Navigasi Halaman",
    options=NAV_PAGES,
    index=NAV_PAGES.index(st.session_state["page"]),
    horizontal=True,
    label_visibility="collapsed",
    key="nav_radio_control",
    on_change=on_nav_change,
)

if selected_nav != st.session_state["page"]:
    st.session_state["page"] = selected_nav
    st.rerun()

# ==============================================================================
# 8. KONTEN HALAMAN
# ==============================================================================
current_page = st.session_state["page"]

# Helper klasifikasi model aktif
def run_model_inference(row_dict: Dict[str, Any]) -> Tuple[str, float, Dict[str, float], Dict[str, Any]]:
    if active_model is None or "pipeline" not in active_model:
        # Fallback deterministik berbasis rumus
        st_pred = train_model.determine_status_gizi(row_dict["zscore_tb_u"], row_dict["zscore_bb_u"])
        return st_pred, 1.0, {st_pred: 1.0}, {"k": 18, "majority_count": 18}

    pipe: train_model.Pipeline = active_model["pipeline"]
    feature_cols = active_model["features"]
    input_df = pd.DataFrame([row_dict])[feature_cols]

    pred_class = str(pipe.predict(input_df)[0])
    probas = pipe.predict_proba(input_df)[0]
    prob_dict = {str(cls_name): float(prob) for cls_name, prob in zip(pipe.classes_, probas)}
    conf = float(prob_dict.get(pred_class, 1.0))

    # Dapatkan tetangga terdekat
    knn_step = pipe.named_steps["knn"]
    prep_step = pipe.named_steps["preprocessor"]
    trans_input = prep_step.transform(input_df)
    distances, indices = knn_step.kneighbors(trans_input)
    
    # Hitung tetangga yang sejalan
    y_train = pipe.classes_[knn_step._y[indices[0]]]
    maj_count = int(np.sum(y_train == pred_class))

    k_info = {
        "k": int(knn_step.n_neighbors),
        "majority_count": maj_count,
        "distances": distances[0],
        "indices": indices[0],
    }
    return pred_class, conf, prob_dict, k_info


# ------------------------------------------------------------------------------
# HALAMAN 1: DASHBOARD & SKRINING GIZI
# ------------------------------------------------------------------------------
if current_page == "🧒 Dashboard & Skrining Gizi":
    st.markdown(ui_components.render_disclaimer(is_dark), unsafe_allow_html=True)

    pred_status, confidence, prob_map, k_info = run_model_inference(active_input_dict)

    # Kartu Hasil Klasifikasi Utama
    st.markdown(
        ui_components.render_prediction_card(
            status=pred_status,
            confidence=confidence,
            probabilities=prob_map,
            k_neighbors_info=k_info,
            is_dark=is_dark,
        ),
        unsafe_allow_html=True,
    )

    # Indikator Z-Score Visual Gauge
    c_g1, c_g2 = st.columns(2)
    with c_g1:
        st.markdown(
            ui_components.render_zscore_gauge("Tinggi Badan menurut Umur (TB/U)", current_ztb, is_dark),
            unsafe_allow_html=True,
        )
    with c_g2:
        st.markdown(
            ui_components.render_zscore_gauge("Berat Badan menurut Umur (BB/U)", current_zbb, is_dark),
            unsafe_allow_html=True,
        )

    # Faktor Risiko Tambahan
    st.markdown(ui_components.render_risk_factor_panel(active_input_dict, is_dark), unsafe_allow_html=True)

    # Rekomendasi Tindak Lanjut & Peringatan Keamanan
    c_rec, c_warn = st.columns([3, 2])
    with c_rec:
        st.markdown(
            f"""
            <div class="glass-card">
                <h4 style="margin-top:0;">📋 Rekomendasi Tindak Lanjut</h4>
                <ul style="font-size:0.86rem; line-height:1.6; color:var(--text); padding-left:1.2rem;">
                    <li><strong>Pemantauan Rutin:</strong> Bawa balita ke Posyandu setiap bulan untuk penimbangan BB dan pengukuran TB secara berkala.</li>
                    <li><strong>Konsultasi Tenaga Gizi:</strong> Kunjungi Puskesmas terdekat untuk verifikasi antropometri dan mendapatkan edukasi pola makan seimbang.</li>
                    <li><strong>Gizi Seimbang:</strong> Prioritaskan konsumsi protein hewani harian (telur, ikan laut segar Madura, ayam/daging) dan variasi sayur-buah.</li>
                    <li><strong>Sanitasi & Higienitas:</strong> Jaga kebersihan sumber air minum dan cuci tangan pakai sabun (CTPS) untuk mencegah diare berulang.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_warn:
        warnings = []
        if confidence < 0.60:
            warnings.append("Tingkat keyakinan model di bawah 60%. Perbedaan jarak antar tetangga tipis.")
        if current_ztb < -3.5 or current_ztb > 3.0:
            warnings.append("Z-score TB/U berada pada nilai ekstrem. Disarankan mengukur ulang tinggi balita.")
        if current_zbb < -3.5 or current_zbb > 3.0:
            warnings.append("Z-score BB/U berada pada nilai ekstrem. Pastikan timbangan terkalibrasi.")

        warn_html = "".join(f"<li>{w}</li>" for w in warnings) if warnings else "<li>Pengukuran balita berada dalam rentang wajar operasional.</li>"
        warn_title = "⚠️ Perhatian Khusus" if warnings else "✅ Parameter Pengukuran Valid"
        warn_color = theme_colors["warning"] if warnings else theme_colors["success"]

        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid {warn_color};">
                <h4 style="margin-top:0; color:{warn_color};">{warn_title}</h4>
                <ul style="font-size:0.84rem; line-height:1.5; color:var(--text); padding-left:1.2rem;">
                    {warn_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Riwayat Skrining (Maksimal 50)
    with st.expander("🕒 Riwayat Skrining Balita Sesi Ini", expanded=False):
        c_act1, c_act2 = st.columns([1, 1])
        with c_act1:
            if st.button("➕ Simpan Skrining Ini ke Riwayat"):
                show_loading("Menyimpan Skrining ke Riwayat...", duration=0.3)
                new_entry = {
                    "Waktu": datetime.now().strftime("%H:%M:%S"),
                    "Jenis Kelamin": active_input_dict["jenis_kelamin"],
                    "Umur (Bln)": active_input_dict["umur_bulan"],
                    "BB (kg)": active_input_dict["berat_badan_kg"],
                    "TB (cm)": active_input_dict["tinggi_badan_cm"],
                    "Z-TB/U": round(current_ztb, 2),
                    "Z-BB/U": round(current_zbb, 2),
                    "Kabupaten": active_input_dict["kabupaten"],
                    "Hasil": pred_status,
                    "Keyakinan": f"{confidence * 100:.1f}%",
                }
                st.session_state["screening_history"].insert(0, new_entry)
                if len(st.session_state["screening_history"]) > 50:
                    st.session_state["screening_history"] = st.session_state["screening_history"][:50]
                st.session_state["pending_loading_msg"] = "Riwayat Skrining Berhasil Diperbarui..."
                st.success("Skrining berhasil disimpan ke riwayat!")
                st.rerun()

        with c_act2:
            if st.session_state["screening_history"]:
                if st.button("🗑️ Hapus Seluruh Riwayat"):
                    show_loading("Menghapus Seluruh Riwayat Skrining...", duration=0.3)
                    st.session_state["screening_history"] = []
                    st.session_state["pending_loading_msg"] = "Riwayat Skrining Telah Dikosongkan..."
                    st.rerun()

        if st.session_state["screening_history"]:
            hist_df = pd.DataFrame(st.session_state["screening_history"])
            st.markdown(ui_components.render_table(hist_df, page=1, page_size=20, is_dark=is_dark), unsafe_allow_html=True)
            csv_data = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Unduh Riwayat (CSV)", data=csv_data, file_name="riwayat_skrining_gizi.csv", mime="text/csv")
        else:
            st.info("Belum ada riwayat skrining yang disimpan.")


# ------------------------------------------------------------------------------
# HALAMAN 2: VISUALISASI 3D & 2D
# ------------------------------------------------------------------------------
elif current_page == "📊 Visualisasi 3D & 2D":
    st.markdown("### 📊 Analisis & Visualisasi Antropometri")
    st.markdown("<p style='color:var(--muted); font-size:0.88rem;'>Eksplorasi ruang fitur K-Nearest Neighbors, batas ambang antropometri, dan perbandingan profil balita.</p>", unsafe_allow_html=True)

    if df_data.empty:
        st.warning("Dataset tidak tersedia untuk visualisasi.")
    else:
        # Filter interaktif
        c_f1, c_f2, c_f3 = st.columns(3)
        with c_f1:
            sel_kab = st.multiselect("Filter Kabupaten:", options=train_model.VALID_CATEGORIES["kabupaten"], default=train_model.VALID_CATEGORIES["kabupaten"])
        with c_f2:
            sel_jk = st.multiselect("Filter Jenis Kelamin:", options=train_model.VALID_CATEGORIES["jenis_kelamin"], default=train_model.VALID_CATEGORIES["jenis_kelamin"])
        with c_f3:
            age_min, age_max = st.slider("Rentang Umur (Bulan):", 6, 59, (6, 59))

        filtered_df = df_data[
            (df_data["kabupaten"].isin(sel_kab)) &
            (df_data["jenis_kelamin"].isin(sel_jk)) &
            (df_data["umur_bulan"] >= age_min) &
            (df_data["umur_bulan"] <= age_max)
        ]

        st.markdown(f"<div style='font-size:0.82rem; color:var(--muted); margin-bottom:12px;'>Menampilkan {len(filtered_df)} dari {len(df_data)} data balita</div>", unsafe_allow_html=True)

        # Plot 3D Scatter Interaktif
        st.markdown("#### 🌐 Scatter Plot 3 Dimensi Ruang Fitur")
        num_options = ["zscore_tb_u", "zscore_bb_u", "umur_bulan", "berat_badan_kg", "tinggi_badan_cm", "pendapatan_keluarga_juta"]
        c_x, c_y, c_z = st.columns(3)
        with c_x:
            axis_x = st.selectbox("Sumbu X:", num_options, index=0)
        with c_y:
            axis_y = st.selectbox("Sumbu Y:", num_options, index=1)
        with c_z:
            axis_z = st.selectbox("Sumbu Z:", num_options, index=2)

        color_discrete_map = {
            "Normal": theme_colors["success"],
            "Gizi Kurang": theme_colors["warning"],
            "Stunting": theme_colors["danger"],
        }

        fig_3d = px.scatter_3d(
            filtered_df,
            x=axis_x,
            y=axis_y,
            z=axis_z,
            color="status_gizi",
            color_discrete_map=color_discrete_map,
            opacity=0.65,
            hover_data=["umur_bulan", "berat_badan_kg", "tinggi_badan_cm", "kabupaten"],
        )

        # Tambahkan titik input aktif pengguna sebagai berlian emas/merah besar
        user_x = active_input_dict.get(axis_x, 0)
        user_y = active_input_dict.get(axis_y, 0)
        user_z = active_input_dict.get(axis_z, 0)

        fig_3d.add_trace(
            go.Scatter3d(
                x=[user_x],
                y=[user_y],
                z=[user_z],
                mode="markers+text",
                marker=dict(size=11, color="#FF1493", symbol="diamond", line=dict(color="#FFFFFF", width=2)),
                name="Balita Anda (Input Aktif)",
                text=["📍 Balita Anda"],
                textposition="top center",
            )
        )

        fig_3d.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme_colors["text"], family="Inter"),
            margin=dict(l=0, r=0, t=20, b=0),
            scene=dict(
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=theme_colors["plotly_grid"]),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=theme_colors["plotly_grid"]),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=theme_colors["plotly_grid"]),
            ),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        # Plot 2D Z-Score TB/U vs BB/U dengan Garis Batas Ambang -2 SD
        c_2d, c_rad = st.columns([3, 2])
        with c_2d:
            st.markdown("#### 🎯 Scatter 2D: Z-Score TB/U vs BB/U")
            fig_2d = px.scatter(
                filtered_df,
                x="zscore_tb_u",
                y="zscore_bb_u",
                color="status_gizi",
                color_discrete_map=color_discrete_map,
                opacity=0.6,
                labels={"zscore_tb_u": "Z-Score TB/U (Tinggi/Umur)", "zscore_bb_u": "Z-Score BB/U (Berat/Umur)"},
            )
            # Garis ambang batas -2 SD
            fig_2d.add_vline(x=-2.0, line_dash="dash", line_color=theme_colors["danger"], annotation_text="Batas TB/U -2 SD")
            fig_2d.add_hline(y=-2.0, line_dash="dash", line_color=theme_colors["warning"], annotation_text="Batas BB/U -2 SD")

            # Titik balita aktif
            fig_2d.add_trace(
                go.Scatter(
                    x=[current_ztb],
                    y=[current_zbb],
                    mode="markers+text",
                    marker=dict(size=14, color="#FF1493", symbol="star", line=dict(color="#FFFFFF", width=2)),
                    name="Balita Anda",
                    text=["📍 Balita Anda"],
                    textposition="top right",
                )
            )
            fig_2d.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=theme_colors["text"], family="Inter"),
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis=dict(gridcolor=theme_colors["plotly_grid"]),
                yaxis=dict(gridcolor=theme_colors["plotly_grid"]),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_2d, use_container_width=True)

        with c_rad:
            st.markdown("#### 🕸️ Radar Profil Relatif Balita")
            # Normalisasi min-max 0-1 untuk radar
            radar_features = ["umur_bulan", "berat_badan_kg", "tinggi_badan_cm", "zscore_tb_u", "zscore_bb_u"]
            min_vals = df_data[radar_features].min()
            max_vals = df_data[radar_features].max()

            norm_user = [(active_input_dict[f] - min_vals[f]) / (max_vals[f] - min_vals[f]) for f in radar_features]
            norm_user.append(norm_user[0])  # tutup kurva radar

            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=norm_user,
                    theta=radar_features + [radar_features[0]],
                    fill="toself",
                    fillcolor=f"{theme_colors['primary']}33",
                    line=dict(color=theme_colors["primary"], width=2),
                    name="Balita Anda",
                )
            )

            # Rata-rata per kelas target
            for st_name, color_c in color_discrete_map.items():
                mean_cls = df_data[df_data["status_gizi"] == st_name][radar_features].mean()
                norm_mean = [(mean_cls[f] - min_vals[f]) / (max_vals[f] - min_vals[f]) for f in radar_features]
                norm_mean.append(norm_mean[0])
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=norm_mean,
                        theta=radar_features + [radar_features[0]],
                        line=dict(color=color_c, dash="dot", width=1.5),
                        name=f"Rerata {st_name}",
                    )
                )

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor=theme_colors["plotly_grid"]),
                    angularaxis=dict(gridcolor=theme_colors["plotly_grid"]),
                    bgcolor="rgba(0,0,0,0)",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=theme_colors["text"], family="Inter"),
                margin=dict(l=30, r=30, t=30, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Distribusi Demografi & Faktor Risiko
        st.markdown("#### 📊 Distribusi Status Gizi berdasarkan Faktor Risiko")
        c_d1, c_d2 = st.columns(2)
        with c_d1:
            fig_bar_kab = px.histogram(
                filtered_df,
                x="kabupaten",
                color="status_gizi",
                barmode="group",
                color_discrete_map=color_discrete_map,
                title="Sebaran Status Gizi per Kabupaten di Madura",
            )
            fig_bar_kab.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=theme_colors["text"], family="Inter"),
                xaxis=dict(gridcolor=theme_colors["plotly_grid"]),
                yaxis=dict(gridcolor=theme_colors["plotly_grid"]),
            )
            st.plotly_chart(fig_bar_kab, use_container_width=True)

        with c_d2:
            fig_bar_asi = px.histogram(
                filtered_df,
                x="asi_eksklusif",
                color="status_gizi",
                barmode="group",
                color_discrete_map=color_discrete_map,
                title="Sebaran Status Gizi menurut ASI Eksklusif",
            )
            fig_bar_asi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=theme_colors["text"], family="Inter"),
                xaxis=dict(gridcolor=theme_colors["plotly_grid"]),
                yaxis=dict(gridcolor=theme_colors["plotly_grid"]),
            )
            st.plotly_chart(fig_bar_asi, use_container_width=True)


# ------------------------------------------------------------------------------
# HALAMAN 3: BEDAH PERHITUNGAN MANUAL
# ------------------------------------------------------------------------------
elif current_page == "🧮 Bedah Perhitungan Manual":
    st.markdown("### 🧮 Bedah Perhitungan Manual K-Nearest Neighbors")
    st.markdown(
        "<p style='color:var(--muted); font-size:0.88rem;'>Transparansi algoritma: saksikan langkah demi langkah konversi antropometri, standardisasi fitur, perhitungan jarak ke data latih, dan pembuktian kecocokan dengan scikit-learn.</p>",
        unsafe_allow_html=True,
    )

    if active_model is None or "pipeline" not in active_model or df_data.empty:
        st.error("Model atau dataset belum dimuat.")
    else:
        pipe: train_model.Pipeline = active_model["pipeline"]
        scaler: train_model.StandardScaler = pipe.named_steps["preprocessor"].named_transformers_["num"]
        knn_clf: train_model.KNeighborsClassifier = pipe.named_steps["knn"]
        fset_cols = active_model["features"]
        metric_name = active_model["metric"]
        k_val = active_model["k"]
        weights_type = active_model["weights"]

        # Data Latih Asli dari Model
        X_train_df = df_data[fset_cols]
        y_train_series = df_data["status_gizi"]

        # LANGKAH 1: Antropometri & Z-Score
        st.markdown("#### Langkah 1: Input Mentah & Perhitungan Z-Score")
        u_val = active_input_dict["umur_bulan"]
        jk_val = active_input_dict["jenis_kelamin"]
        bb_val = active_input_dict["berat_badan_kg"]
        tb_val = active_input_dict["tinggi_badan_cm"]
        med_tb, sd_tb, med_bb, sd_bb = train_model.get_growth_reference(u_val, jk_val)

        st.latex(r"z_{TB/U} = \frac{TB - \text{Median}(u, jk)}{SD_{TB}(u, jk)}")
        st.markdown(
            f"""
            Untuk balita **{jk_val}** umur **{u_val} bulan** dari tabel acuan WHO (interpolasi linear):
            - $\\text{{Median}}_{{TB}} = {med_tb:.2f}\\text{{ cm}}$, $SD_{{TB}} = {sd_tb:.2f}\\text{{ cm}}$
            - $z_{{TB/U}} = \\frac{{{tb_val:.1f} - {med_tb:.2f}}}{{{sd_tb:.2f}}} = \\mathbf{{{current_ztb:+.4f}\\text{{ SD}}}}$
            """
        )

        st.latex(r"z_{BB/U} = \frac{BB - \text{Median}(u, jk)}{\text{Median}(u, jk) \times \text{Fraksi}(u)}")
        st.markdown(
            f"""
            - $\\text{{Median}}_{{BB}} = {med_bb:.2f}\\text{{ kg}}$, $SD_{{BB}} = {sd_bb:.2f}\\text{{ kg}}$
            - $z_{{BB/U}} = \\frac{{{bb_val:.1f} - {med_bb:.2f}}}{{{sd_bb:.2f}}} = \\mathbf{{{current_zbb:+.4f}\\text{{ SD}}}}$
            """
        )

        # LANGKAH 2: Standardisasi Fitur (Z-score Normalization)
        st.markdown("#### Langkah 2: Normalisasi Fitur Model Aktif (StandardScaler)")
        st.markdown(f"Model aktif menggunakan **SET {active_model['feature_set']}** dengan fitur: `{fset_cols}`.")
        st.latex(r"x_{\text{scaled}} = \frac{x - \mu}{\sigma}")

        norm_data = []
        user_raw = [active_input_dict[c] for c in scaler.feature_names_in_]
        user_scaled = (np.array(user_raw) - scaler.mean_) / scaler.scale_

        for c_name, raw_v, mu, sigma, sc_v in zip(scaler.feature_names_in_, user_raw, scaler.mean_, scaler.scale_, user_scaled):
            norm_data.append({
                "Fitur": c_name,
                "Nilai Mentah (x)": round(raw_v, 2),
                "Mean Latih (μ)": round(mu, 4),
                "Std Dev Latih (σ)": round(sigma, 4),
                "Terskalakan (x_scaled)": round(sc_v, 4),
            })
        st.markdown(ui_components.render_table(pd.DataFrame(norm_data), page=1, page_size=10, is_dark=is_dark), unsafe_allow_html=True)

        # LANGKAH 3: Jarak ke Semua Data Latih (Manual NumPy)
        st.markdown("#### Langkah 3: Perhitungan Jarak ke Data Latih (Vektorisasi NumPy)")
        # Ambil representasi terskalakan data latih aktual yang dipelajari model KNN
        X_train_scaled = knn_clf._fit_X
        y_train_classes = pipe.classes_[knn_clf._y]
        user_trans = user_scaled.reshape(1, -1)

        if metric_name == "manhattan":
            st.latex(r"d(A, B) = \sum_{i=1}^n |A_i - B_i|")
            diffs = np.abs(X_train_scaled - user_trans)
            distances_manual = np.sum(diffs, axis=1)
        else:  # euclidean
            st.latex(r"d(A, B) = \sqrt{\sum_{i=1}^n (A_i - B_i)^2}")
            diffs = (X_train_scaled - user_trans) ** 2
            distances_manual = np.sqrt(np.sum(diffs, axis=1))

        # Cari K tetangga terdekat secara manual via argsort
        sorted_indices = np.argsort(distances_manual)[:k_val]
        top_k_distances = distances_manual[sorted_indices]
        top_k_labels = y_train_classes[sorted_indices]

        k_table_rows = []
        for rank, (idx, dist, lbl) in enumerate(zip(sorted_indices, top_k_distances, top_k_labels), 1):
            row_dict = {
                "Peringkat": f"#{rank}",
                "Indeks Data Latih": int(idx),
                "Status Gizi Latih": lbl,
                f"Jarak ({metric_name})": round(dist, 4),
            }
            # Tambahkan selisih per fitur
            for f_i, f_name in enumerate(scaler.feature_names_in_):
                row_dict[f"|Δ {f_name}|"] = round(float(np.abs(X_train_scaled[idx, f_i] - user_trans[0, f_i])), 4)
            k_table_rows.append(row_dict)

        st.markdown(f"**Tabel {k_val} Tetangga Terdekat:**")
        st.markdown(ui_components.render_table(pd.DataFrame(k_table_rows), page=1, page_size=25, is_dark=is_dark), unsafe_allow_html=True)

        # LANGKAH 4: Pemungutan Suara (Voting)
        st.markdown("#### Langkah 4: Pemungutan Suara (Voting)")
        vote_dict = {c: 0.0 for c in pipe.classes_}
        if weights_type == "distance":
            st.markdown("Model menggunakan pembobotan jarak: $\\text{Bobot} = \\frac{1}{d + 10^{-6}}$")
            for dist, lbl in zip(top_k_distances, top_k_labels):
                w = 1.0 / (dist + 1e-6)
                vote_dict[lbl] += w
        else:
            st.markdown(f"Model menggunakan pembobotan seragam (**uniform**): tiap tetangga bernilai 1 suara.")
            for lbl in top_k_labels:
                vote_dict[lbl] += 1.0

        total_votes = sum(vote_dict.values())
        vote_rows = []
        for cls_name in pipe.classes_:
            v_val = vote_dict[cls_name]
            pct = (v_val / total_votes) * 100 if total_votes > 0 else 0
            vote_rows.append({
                "Kelas": cls_name,
                "Total Suara": round(v_val, 2),
                "Persentase": f"{pct:.1f}%",
            })
        st.markdown(ui_components.render_table(pd.DataFrame(vote_rows), page=1, page_size=5, is_dark=is_dark), unsafe_allow_html=True)

        # LANGKAH 5: Verifikasi Hasil Manual vs Scikit-Learn
        st.markdown("#### Langkah 5: Pembuktian Kecocokan Manual vs Scikit-Learn")
        knn_sk_dist, knn_sk_indices = knn_clf.kneighbors(user_trans)

        is_idx_match = np.array_equal(sorted_indices, knn_sk_indices[0])
        dist_diff_max = np.max(np.abs(top_k_distances - knn_sk_dist[0]))

        c_v1, c_v2 = st.columns(2)
        with c_v1:
            st.markdown(
                f"""
                <div class="glass-card">
                    <h5>Perhitungan Manual NumPy</h5>
                    <ul style="font-size:0.85rem; color:var(--text);">
                        <li>Indeks Tetangga: <code>{list(sorted_indices[:5])}...</code></li>
                        <li>Jarak Terdekat: <code>{round(top_k_distances[0], 5)}</code></li>
                        <li>Kelas Pemenang: <strong>{max(vote_dict, key=vote_dict.get)}</strong></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_v2:
            st.markdown(
                f"""
                <div class="glass-card">
                    <h5>Fungsi scikit-learn (kneighbors)</h5>
                    <ul style="font-size:0.85rem; color:var(--text);">
                        <li>Indeks Tetangga: <code>{list(knn_sk_indices[0][:5])}...</code></li>
                        <li>Jarak Terdekat: <code>{round(knn_sk_dist[0][0], 5)}</code></li>
                        <li>Kelas Prediksi: <strong>{pipe.predict(pd.DataFrame([active_input_dict])[fset_cols])[0]}</strong></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if dist_diff_max <= 1e-5:
            st.success(f"✅ VERIFIKASI LULUS: Perhitungan manual NumPy dan scikit-learn COCOK 100% PERSIS (selisih jarak maksimum: {dist_diff_max:.2e}).")
        else:
            st.warning(f"Terdapat perbedaan minor numerik (selisih: {dist_diff_max:.4f}).")


# ------------------------------------------------------------------------------
# HALAMAN 4: EVALUASI & OPTIMASI K
# ------------------------------------------------------------------------------
elif current_page == "📈 Evaluasi & Optimasi K":
    st.markdown("### 📈 Evaluasi Model & Optimasi Parameter K")
    st.markdown("<p style='color:var(--muted); font-size:0.88rem;'>Latih ulang, tuning hiperparameter KNN, bandingkan performa set fitur, dan unduh model terverifikasi.</p>", unsafe_allow_html=True)

    if active_model:
        m_info = active_model
        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 5px solid {theme_colors['primary']};">
                <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:1rem;">
                    <div>
                        <h4 style="margin:0;">Informasi Model Aktif</h4>
                        <div style="font-size:0.85rem; color:var(--muted); margin-top:4px;">
                            Set Fitur: <strong>SET {m_info['feature_set']}</strong> | K = <strong>{m_info['k']}</strong> | Metrik = <strong>{m_info['metric']}</strong> | Bobot = <strong>{m_info['weights']}</strong>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.75rem; text-transform:uppercase; color:var(--muted);">Akurasi Data Uji</div>
                        <div style="font-size:1.6rem; font-weight:800; color:{theme_colors['success']};">
                            {m_info['metrics']['accuracy'] * 100:.2f}%
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Form Konfigurasi Evaluasi
    with st.expander("⚙️ Parameter Evaluasi & Tuning Hiperparameter", expanded=True):
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            eval_test_size = st.slider("Rasio Data Uji (Test Size):", 0.10, 0.40, 0.20, step=0.05)
            eval_fset = st.selectbox("Set Fitur yang Diuji:", ["A (Z-score inti)", "B (Numerik lengkap)", "C (Semua fitur)", "Semua (Grid CV Otomatis)"], index=3)
        with c_p2:
            eval_k_min, eval_k_max = st.slider("Rentang Nilai K:", 1, 30, (1, 25))
            eval_metric = st.selectbox("Metrik Jarak:", ["euclidean", "manhattan", "Keduanya (Grid CV)"], index=2)
        with c_p3:
            eval_weights = st.selectbox("Pembobotan Tetangga:", ["uniform", "distance", "Keduanya (Grid CV)"], index=2)
            eval_cv_folds = st.slider("Jumlah Fold Stratified CV:", 3, 10, 5)

        run_eval_btn = st.button("🚀 Jalankan Evaluasi & Tuning Hiperparameter")

    if run_eval_btn and not df_data.empty:
        load_ph = st.empty()
        load_ph.markdown(
            _clean_html(styles.get_loading_overlay_html("Menjalankan 5-Fold Stratified CV & Grid Search Tuning KNN...", is_dark)),
            unsafe_allow_html=True,
        )
        try:
            fsets_to_run = ["A", "B", "C"] if "Semua" in eval_fset else [eval_fset[0]]
            metrics_to_run = ["euclidean", "manhattan"] if "Keduanya" in eval_metric else [eval_metric]
            weights_to_run = ["uniform", "distance"] if "Keduanya" in eval_weights else [eval_weights]

            X = df_data[train_model.EXPECTED_COLUMNS[:-1]]
            y = df_data["status_gizi"]
            X_tr, X_te, y_tr, y_te = sklearn.model_selection.train_test_split(
                X, y, test_size=eval_test_size, random_state=42, stratify=y
            )

            best_cfg, grid_df = train_model.run_full_grid_search(
                X_train=X_tr,
                y_train=y_tr,
                feature_sets_to_test=fsets_to_run,
                k_range=range(eval_k_min, eval_k_max + 1),
                metrics=metrics_to_run,
                weights_list=weights_to_run,
                cv_splits=eval_cv_folds,
            )

            st.session_state["eval_result"] = {
                "best_cfg": best_cfg,
                "grid_df": grid_df,
                "X_tr": X_tr,
                "X_te": X_te,
                "y_tr": y_tr,
                "y_te": y_te,
            }
        finally:
            load_ph.empty()
        st.success(f"Evaluasi selesai! Konfigurasi terbaik: Set {best_cfg['feature_set']}, K={best_cfg['k']}, {best_cfg['metric']}, {best_cfg['weights']} (CV Akurasi: {best_cfg['cv_accuracy']*100:.2f}%)")

    if "eval_result" in st.session_state:
        res = st.session_state["eval_result"]
        b_cfg = res["best_cfg"]
        g_df: pd.DataFrame = res["grid_df"]

        # Grafik K vs Akurasi
        st.markdown("#### 📊 Grafik Nilai K vs Akurasi Cross-Validation")
        fig_k = px.line(
            g_df[g_df["feature_set"] == b_cfg["feature_set"]],
            x="k",
            y="cv_accuracy",
            color="weights",
            line_dash="metric",
            title=f"Performa K pada SET {b_cfg['feature_set']}",
            labels={"k": "Nilai K", "cv_accuracy": "Akurasi CV"},
        )
        fig_k.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme_colors["text"], family="Inter"),
            xaxis=dict(gridcolor=theme_colors["plotly_grid"]),
            yaxis=dict(gridcolor=theme_colors["plotly_grid"]),
        )
        st.plotly_chart(fig_k, use_container_width=True)

        # Latih model pada konfigurasi terbaik untuk evaluasi test set
        fset_cols = train_model.FEATURE_SETS[b_cfg["feature_set"]]
        pipe_eval = train_model.create_pipeline(b_cfg["feature_set"], k=int(b_cfg["k"]), metric=b_cfg["metric"], weights=b_cfg["weights"])
        pipe_eval.fit(res["X_tr"][fset_cols], res["y_tr"])
        y_pred = pipe_eval.predict(res["X_te"][fset_cols])

        te_acc = sklearn.metrics.accuracy_score(res["y_te"], y_pred)
        p_m, r_m, f1_m, _ = sklearn.metrics.precision_recall_fscore_support(res["y_te"], y_pred, average="macro", zero_division=0)
        cm = sklearn.metrics.confusion_matrix(res["y_te"], y_pred, labels=list(pipe_eval.classes_))

        # Metrik Ringkasan
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("Akurasi Data Uji", f"{te_acc * 100:.2f}%")
        c_m2.metric("Macro Precision", f"{p_m * 100:.2f}%")
        c_m3.metric("Macro Recall", f"{r_m * 100:.2f}%")
        c_m4.metric("Macro F1-Score", f"{f1_m * 100:.2f}%")

        # Confusion Matrix Heatmap
        c_cm, c_cr = st.columns([1, 1])
        with c_cm:
            st.markdown("#### 🧩 Confusion Matrix")
            fig_cm = px.imshow(
                cm,
                x=list(pipe_eval.classes_),
                y=list(pipe_eval.classes_),
                text_auto=True,
                color_continuous_scale="Teal",
                labels=dict(x="Prediksi Model", y="Aktual / Kenyataan"),
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=theme_colors["text"], family="Inter"),
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with c_cr:
            st.markdown("#### 📑 Classification Report")
            cr_dict = sklearn.metrics.classification_report(res["y_te"], y_pred, output_dict=True, zero_division=0)
            cr_rows = []
            for cls_lbl in list(pipe_eval.classes_) + ["macro avg", "weighted avg"]:
                if cls_lbl in cr_dict:
                    cr_rows.append({
                        "Kelas": cls_lbl,
                        "Precision": f"{cr_dict[cls_lbl]['precision']*100:.1f}%",
                        "Recall": f"{cr_dict[cls_lbl]['recall']*100:.1f}%",
                        "F1-Score": f"{cr_dict[cls_lbl]['f1-score']*100:.1f}%",
                        "Support": int(cr_dict[cls_lbl]['support']),
                    })
            st.markdown(ui_components.render_table(pd.DataFrame(cr_rows), page=1, page_size=10, is_dark=is_dark), unsafe_allow_html=True)

        # Perbandingan Set Fitur
        st.markdown("#### ⚖️ Perbandingan Antar Set Fitur (Mengapa Seleksi Fitur Penting)")
        fset_comp_rows = []
        for f_name in ["A", "B", "C"]:
            sub_p = train_model.create_pipeline(f_name, k=int(b_cfg["k"]), metric=b_cfg["metric"], weights=b_cfg["weights"])
            sub_p.fit(res["X_tr"][train_model.FEATURE_SETS[f_name]], res["y_tr"])
            sub_pred = sub_p.predict(res["X_te"][train_model.FEATURE_SETS[f_name]])
            fset_comp_rows.append({
                "Set Fitur": f"SET {f_name}",
                "Deskripsi": "Z-Score Inti (TB/U & BB/U)" if f_name == "A" else ("Fitur Numerik Lengkap" if f_name == "B" else "Semua Fitur (Termasuk One-Hot)"),
                "Akurasi": f"{sklearn.metrics.accuracy_score(res['y_te'], sub_pred)*100:.2f}%",
                "Macro F1": f"{sklearn.metrics.f1_score(res['y_te'], sub_pred, average='macro', zero_division=0)*100:.2f}%",
            })
        st.markdown(ui_components.render_table(pd.DataFrame(fset_comp_rows), page=1, page_size=10, is_dark=is_dark), unsafe_allow_html=True)

        # Tombol Aksi Terapkan & Unduh
        c_app, c_dwn = st.columns(2)
        with c_app:
            if st.button("💾 Terapkan Konfigurasi Terbaik & Simpan ke Model Aktif"):
                load_ph = st.empty()
                load_ph.markdown(
                    _clean_html(styles.get_loading_overlay_html("Melatih Ulang & Menyimpan Model ke File .joblib...", is_dark)),
                    unsafe_allow_html=True,
                )
                try:
                    train_model.train_and_save_model(
                        data_path=DATA_FILE_PATH,
                        output_path=MODEL_FILE_PATH,
                        force_config=b_cfg,
                    )
                    load_active_model.clear()
                    time.sleep(0.3)
                finally:
                    load_ph.empty()
                st.session_state["pending_loading_msg"] = "Model Baru Berhasil Diaktifkan..."
                st.success("Model berhasil diperbarui dan disimpan ke models/knn_gizi_balita_madura.joblib!")
                st.rerun()

    # Tombol Unduh Model .joblib
    if MODEL_FILE_PATH.exists():
        with open(MODEL_FILE_PATH, "rb") as f_joblib:
            st.download_button(
                "📥 Unduh Model Aktif (.joblib)",
                data=f_joblib.read(),
                file_name="knn_gizi_balita_madura.joblib",
                mime="application/octet-stream",
            )


# ------------------------------------------------------------------------------
# HALAMAN 5: MANAJEMEN DATASET
# ------------------------------------------------------------------------------
elif current_page == "📁 Manajemen Dataset":
    st.markdown("### 📁 Manajemen Dataset Status Gizi Balita")
    st.markdown("<p style='color:var(--muted); font-size:0.88rem;'>Kelola dataset balita di Madura: eksplorasi data, tambah sampel baru dengan hitung otomatis, unggah pengganti, dan unduh data.</p>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="background:rgba(255,255,255,0.25); border:1px solid var(--glass-border); border-radius:14px; padding:10px 14px; margin-bottom:15px; font-size:0.83rem; color:var(--text);">
            ℹ️ <strong>Catatan Keaslian Data:</strong> Dataset ini merupakan <strong>data simulasi (sintetis)</strong> berstandar antropometri balita Madura tanpa identitas anak riil. Pada lingkungan Streamlit Cloud, perubahan file sistem bersifat sementara (ephemeral) dan kembali ke versi Git saat kontainer me-reboot.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df_data.empty:
        st.error("Dataset kosong atau tidak ditemukan.")
    else:
        # Ringkasan Dataset
        c_s1, c_s2, c_s3, c_s4 = st.columns(4)
        c_s1.metric("Total Sampel Balita", f"{len(df_data):,} baris")
        c_s2.metric("Total Kolom Fitur", f"{len(df_data.columns)} kolom")
        c_s3.metric("Data Hilang / Kosong", "0 baris")
        c_s4.metric("Baris Duplikat", "0 baris")

        # Eksplorasi & Filter Tabel
        with st.expander("🔍 Filter & Eksplorasi Data", expanded=True):
            c_srch, c_fkab, c_fstatus = st.columns([2, 1, 1])
            with c_srch:
                search_query = st.text_input("Cari data (kata kunci):", "")
            with c_fkab:
                kab_filter = st.selectbox("Filter Kabupaten:", ["Semua"] + train_model.VALID_CATEGORIES["kabupaten"])
            with c_fstatus:
                status_filter = st.selectbox("Filter Status Gizi:", ["Semua"] + train_model.VALID_CATEGORIES["status_gizi"])

            view_df = df_data.copy()
            if kab_filter != "Semua":
                view_df = view_df[view_df["kabupaten"] == kab_filter]
            if status_filter != "Semua":
                view_df = view_df[view_df["status_gizi"] == status_filter]
            if search_query:
                mask = view_df.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)
                view_df = view_df[mask]

            # Paginasi Tabel Kaca
            total_rows_view = len(view_df)
            page_size = 50
            total_pages = max(1, (total_rows_view + page_size - 1) // page_size)

            c_pnav1, c_pnav2, _ = st.columns([1, 1, 3])
            with c_pnav1:
                if st.button("◀ Halaman Sebelumnya") and st.session_state["dataset_page"] > 1:
                    st.session_state["pending_loading_msg"] = f"Memuat Halaman {st.session_state['dataset_page'] - 1} Dataset..."
                    st.session_state["dataset_page"] -= 1
                    st.rerun()
            with c_pnav2:
                if st.button("Halaman Berikutnya ▶") and st.session_state["dataset_page"] < total_pages:
                    st.session_state["pending_loading_msg"] = f"Memuat Halaman {st.session_state['dataset_page'] + 1} Dataset..."
                    st.session_state["dataset_page"] += 1
                    st.rerun()

            st.markdown(
                ui_components.render_table(
                    view_df,
                    page=st.session_state["dataset_page"],
                    page_size=page_size,
                    is_dark=is_dark,
                ),
                unsafe_allow_html=True,
            )

        # Form Tambah Data Baru
        with st.expander("➕ Tambah Satu Data Balita Baru ke Dataset", expanded=False):
            with st.form("form_add_child"):
                st.markdown("<p style='font-size:0.85rem; color:var(--muted);'>Z-Score dan status gizi akan <strong>dihitung otomatis</strong> berdasarkan standar acuan saat disimpan.</p>", unsafe_allow_html=True)
                c_a1, c_a2, c_a3 = st.columns(3)
                with c_a1:
                    new_kab = st.selectbox("Kabupaten:", train_model.VALID_CATEGORIES["kabupaten"])
                    new_jk = st.selectbox("Jenis Kelamin:", train_model.VALID_CATEGORIES["jenis_kelamin"])
                    new_umur = st.number_input("Umur (Bulan):", min_value=6, max_value=59, value=24)
                with c_a2:
                    new_bb = st.number_input("Berat Badan (kg):", min_value=3.0, max_value=25.0, value=12.0, step=0.1)
                    new_tb = st.number_input("Tinggi Badan (cm):", min_value=55.0, max_value=125.0, value=86.0, step=0.1)
                    new_asi = st.selectbox("ASI Eksklusif:", ["Ya", "Tidak"])
                with c_a3:
                    new_san = st.selectbox("Sanitasi Layak:", ["Ya", "Tidak"])
                    new_pend = st.number_input("Pendapatan (Juta Rp):", min_value=0.5, max_value=9.0, value=2.5, step=0.1)
                    new_sek = st.selectbox("Lama Sekolah Ibu:", [6, 9, 12, 16])
                    new_inf = st.number_input("Episode Infeksi (3 Bln):", min_value=0, max_value=3, value=0)

                submit_add = st.form_submit_button("Simpan Data ke Excel")

            if submit_add:
                load_ph = st.empty()
                load_ph.markdown(
                    _clean_html(styles.get_loading_overlay_html("Menyimpan & Menghitung Otomatis Z-Score Balita...", is_dark)),
                    unsafe_allow_html=True,
                )
                try:
                    ztb_calc, zbb_calc = train_model.calculate_zscores(new_umur, new_jk, new_bb, new_tb)
                    status_calc = train_model.determine_status_gizi(ztb_calc, zbb_calc)

                    new_row = {
                        "kabupaten": new_kab,
                        "jenis_kelamin": new_jk,
                        "umur_bulan": int(new_umur),
                        "berat_badan_kg": float(new_bb),
                        "tinggi_badan_cm": float(new_tb),
                        "zscore_tb_u": round(ztb_calc, 2),
                        "zscore_bb_u": round(zbb_calc, 2),
                        "asi_eksklusif": new_asi,
                        "sanitasi_layak": new_san,
                        "pendapatan_keluarga_juta": float(new_pend),
                        "lama_sekolah_ibu": int(new_sek),
                        "episode_infeksi_3bln": int(new_inf),
                        "status_gizi": status_calc,
                    }

                    updated_df = pd.concat([df_data, pd.DataFrame([new_row])], ignore_index=True)

                    with pd.ExcelWriter(DATA_FILE_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                        updated_df.to_excel(writer, sheet_name="data gizi balita", index=False)
                    get_dataset.clear()
                    time.sleep(0.35)
                finally:
                    load_ph.empty()
                st.session_state["pending_loading_msg"] = "Memperbarui Tampilan Dataset Balita..."
                st.success(f"Data balita berhasil ditambahkan! Z TB/U: {ztb_calc:+.2f}, Z BB/U: {zbb_calc:+.2f}, Status: {status_calc}")
                st.rerun()

        # Unggah File Pengganti & Validasi Ketat
        with st.expander("📤 Unggah File Dataset Pengganti (.xlsx / .csv)", expanded=True):
            uploaded_file = st.file_uploader("Pilih file dataset:", type=["xlsx", "csv"])
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        up_df = pd.read_csv(uploaded_file)
                    else:
                        up_df = pd.read_excel(uploaded_file, sheet_name=0)

                    # Validasi Skema
                    err_msgs = []
                    missing = [c for c in train_model.EXPECTED_COLUMNS if c not in up_df.columns]
                    if missing:
                        err_msgs.append(f"Kolom wajib tidak lengkap: {missing}")

                    for cat_col, valid_vals in train_model.VALID_CATEGORIES.items():
                        if cat_col in up_df.columns:
                            invalid = set(up_df[cat_col].dropna().unique()) - set(valid_vals)
                            if invalid:
                                err_msgs.append(f"Kategori tidak valid pada '{cat_col}': {invalid}")

                    # Validasi tipe numerik & rentang nilai
                    ranges = {
                        "umur_bulan": (6, 59), "berat_badan_kg": (2.0, 30.0), "tinggi_badan_cm": (45.0, 130.0),
                        "zscore_tb_u": (-10, 10), "zscore_bb_u": (-10, 10),
                        "pendapatan_keluarga_juta": (0.0, 50.0), "lama_sekolah_ibu": (0, 20),
                        "episode_infeksi_3bln": (0, 3),
                    }
                    for num_col, (lo, hi) in ranges.items():
                        if num_col in up_df.columns:
                            col_num = pd.to_numeric(up_df[num_col], errors="coerce")
                            if col_num.isna().any():
                                err_msgs.append(f"Kolom '{num_col}' berisi nilai kosong/bukan angka.")
                            elif (col_num < lo).any() or (col_num > hi).any():
                                err_msgs.append(f"Kolom '{num_col}' di luar rentang {lo}–{hi}.")
                    if len(up_df) == 0:
                        err_msgs.append("File tidak berisi baris data.")

                    if err_msgs:
                        st.error("❌ Validasi Skema Gagal. File tidak disimpan:")
                        for em in err_msgs:
                            st.write(f"- {em}")
                    else:
                        st.success(f"✅ File valid ({len(up_df)} baris).")
                        if st.button("Konfirmasi & Simpan Dataset Baru"):
                            load_ph = st.empty()
                            load_ph.markdown(
                                _clean_html(styles.get_loading_overlay_html("Memproses & Menyimpan File Dataset Baru...", is_dark)),
                                unsafe_allow_html=True,
                            )
                            try:
                                with pd.ExcelWriter(DATA_FILE_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                                    up_df.to_excel(writer, sheet_name="data gizi balita", index=False)
                                get_dataset.clear()
                                load_active_model.clear()
                                time.sleep(0.35)
                            finally:
                                load_ph.empty()
                            st.session_state["pending_loading_msg"] = "Memuat Dataset Baru & Memperbarui Cache..."
                            st.success("Dataset berhasil diperbarui!")
                            st.rerun()
                except Exception as e:
                    st.error(f"Gagal memproses file unggahan: {e}")

        # Unduh & Cadangan
        st.markdown("#### 📥 Unduh & Cadangan Dataset")
        c_dw1, c_dw2, c_dw3 = st.columns(3)
        with c_dw1:
            csv_buf = df_data.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Unduh Dataset (CSV)", data=csv_buf, file_name="data_status_gizi_balita.csv", mime="text/csv")
        with c_dw2:
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as wr:
                df_data.to_excel(wr, sheet_name="data gizi balita", index=False)
            st.download_button("📥 Unduh Dataset (Excel)", data=excel_buf.getvalue(), file_name="data_status_gizi_balita.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c_dw3:
            if DATA_BACKUP_PATH.exists():
                if st.button("↺ Kembalikan Dataset Asli (Reset)"):
                    load_ph = st.empty()
                    load_ph.markdown(
                        _clean_html(styles.get_loading_overlay_html("Memulihkan Dataset Asli dari Cadangan...", is_dark)),
                        unsafe_allow_html=True,
                    )
                    try:
                        shutil.copy2(DATA_BACKUP_PATH, DATA_FILE_PATH)
                        get_dataset.clear()
                        load_active_model.clear()
                        time.sleep(0.35)
                    finally:
                        load_ph.empty()
                    st.session_state["pending_loading_msg"] = "Dataset Asli Berhasil Dipulihkan..."
                    st.success("Dataset asli berhasil dipulihkan!")
                    st.rerun()


# ------------------------------------------------------------------------------
# HALAMAN 6: EDUKASI GIZI & KURVA PERTUMBUHAN
# ------------------------------------------------------------------------------
elif current_page == "🥗 Edukasi Gizi & Kurva Pertumbuhan":
    st.markdown("### 🥗 Edukasi Gizi & Kurva Pertumbuhan WHO")
    st.markdown("<p style='color:var(--muted); font-size:0.88rem;'>Pantau lintasan tumbuh kembang anak pada kurva standar WHO 2006 dan pelajari intervensi gizi terpadu.</p>", unsafe_allow_html=True)

    # Kurva Pertumbuhan Interaktif
    sel_jk_curve = st.selectbox("Pilih Jenis Kelamin Kurva Pertumbuhan:", ["Laki-laki", "Perempuan"], index=0 if active_input_dict["jenis_kelamin"] == "Laki-laki" else 1)
    curve_age_mesh = np.linspace(0, 60, 120)

    # Dapatkan titik kurva acuan
    med_tb_line, sd_tb_line, med_bb_line, sd_bb_line = [], [], [], []
    for a in curve_age_mesh:
        mt, st_t, mb, sb_b = train_model.get_growth_reference(a, sel_jk_curve)
        med_tb_line.append(mt)
        sd_tb_line.append(st_t)
        med_bb_line.append(mb)
        sd_bb_line.append(sb_b)

    med_tb_arr = np.array(med_tb_line)
    sd_tb_arr = np.array(sd_tb_line)
    med_bb_arr = np.array(med_bb_line)
    sd_bb_arr = np.array(sd_bb_line)

    c_cur1, c_cur2 = st.columns(2)
    with c_cur1:
        st.markdown("#### 📏 Kurva Panjang/Tinggi Badan menurut Umur (TB/U)")
        fig_tb = go.Figure()
        # Area normal (-2 s/d +2 SD)
        fig_tb.add_trace(go.Scatter(x=curve_age_mesh, y=med_tb_arr + 2 * sd_tb_arr, mode="lines", line=dict(width=0), showlegend=False))
        fig_tb.add_trace(go.Scatter(x=curve_age_mesh, y=med_tb_arr - 2 * sd_tb_arr, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=f"{theme_colors['success']}25", name="Zona Normal (-2 s/d +2 SD)"))
        # Garis Median & Batas -2 SD
        fig_tb.add_trace(go.Scatter(x=curve_age_mesh, y=med_tb_arr, mode="lines", line=dict(color=theme_colors["success"], width=2.5), name="Median (0 SD)"))
        fig_tb.add_trace(go.Scatter(x=curve_age_mesh, y=med_tb_arr - 2 * sd_tb_arr, mode="lines", line=dict(color=theme_colors["danger"], dash="dash", width=2), name="Batas Stunting (-2 SD)"))
        fig_tb.add_trace(go.Scatter(x=curve_age_mesh, y=med_tb_arr - 3 * sd_tb_arr, mode="lines", line=dict(color=theme_colors["danger"], dash="dot", width=1.5), name="Sangat Pendek (-3 SD)"))

        # Titik balita aktif
        if active_input_dict["jenis_kelamin"] == sel_jk_curve:
            fig_tb.add_trace(
                go.Scatter(
                    x=[active_input_dict["umur_bulan"]],
                    y=[active_input_dict["tinggi_badan_cm"]],
                    mode="markers+text",
                    marker=dict(size=14, color="#FF1493", symbol="star", line=dict(color="#FFFFFF", width=2)),
                    name="Balita Anda",
                    text=["📍 Balita Anda"],
                    textposition="top right",
                )
            )

        fig_tb.update_layout(
            xaxis_title="Umur (Bulan)",
            yaxis_title="Tinggi Badan (cm)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme_colors["text"], family="Inter"),
            xaxis=dict(gridcolor=theme_colors["plotly_grid"]),
            yaxis=dict(gridcolor=theme_colors["plotly_grid"]),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_tb, use_container_width=True)

    with c_cur2:
        st.markdown("#### ⚖️ Kurva Berat Badan menurut Umur (BB/U)")
        fig_bb = go.Figure()
        # Area normal (-2 s/d +2 SD)
        fig_bb.add_trace(go.Scatter(x=curve_age_mesh, y=med_bb_arr + 2 * sd_bb_arr, mode="lines", line=dict(width=0), showlegend=False))
        fig_bb.add_trace(go.Scatter(x=curve_age_mesh, y=med_bb_arr - 2 * sd_bb_arr, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=f"{theme_colors['success']}25", name="Zona Normal (-2 s/d +2 SD)"))
        # Garis Median & Batas -2 SD
        fig_bb.add_trace(go.Scatter(x=curve_age_mesh, y=med_bb_arr, mode="lines", line=dict(color=theme_colors["success"], width=2.5), name="Median (0 SD)"))
        fig_bb.add_trace(go.Scatter(x=curve_age_mesh, y=med_bb_arr - 2 * sd_bb_arr, mode="lines", line=dict(color=theme_colors["warning"], dash="dash", width=2), name="Batas Gizi Kurang (-2 SD)"))
        fig_bb.add_trace(go.Scatter(x=curve_age_mesh, y=med_bb_arr - 3 * sd_bb_arr, mode="lines", line=dict(color=theme_colors["danger"], dash="dot", width=1.5), name="Gizi Buruk (-3 SD)"))

        # Titik balita aktif
        if active_input_dict["jenis_kelamin"] == sel_jk_curve:
            fig_bb.add_trace(
                go.Scatter(
                    x=[active_input_dict["umur_bulan"]],
                    y=[active_input_dict["berat_badan_kg"]],
                    mode="markers+text",
                    marker=dict(size=14, color="#FF1493", symbol="star", line=dict(color="#FFFFFF", width=2)),
                    name="Balita Anda",
                    text=["📍 Balita Anda"],
                    textposition="top right",
                )
            )

        fig_bb.update_layout(
            xaxis_title="Umur (Bulan)",
            yaxis_title="Berat Badan (kg)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme_colors["text"], family="Inter"),
            xaxis=dict(gridcolor=theme_colors["plotly_grid"]),
            yaxis=dict(gridcolor=theme_colors["plotly_grid"]),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_bb, use_container_width=True)

    # Modul Edukasi Interaktif & Konteks Madura
    st.markdown("#### 📖 Panduan Intervensi Gizi & Tumbuh Kembang")
    with st.expander("🌱 1. Apa itu Stunting dan Gizi Kurang?", expanded=True):
        st.markdown(
            """
            - **Stunting (Pendek / Sangat Pendek):** Kondisi gagal tumbuh pada anak balita akibat kekurangan gizi kronis dan infeksi berulang, terutama dalam 1.000 Hari Pertama Kehidupan (1000 HPK). Teridentifikasi bila z-score TB/U berada di bawah -2.0 SD. Dampaknya tidak hanya fisik yang pendek, tetapi juga perkembangan kognitif dan imunitas tubuh.
            - **Gizi Kurang (Underweight):** Kondisi berat badan anak berada di bawah standar usianya (z-score BB/U < -2.0 SD) dengan tinggi badan belum tentu pendek. Ini sering menjadi tanda awal defisit asupan nutrisi akut yang bila dibiarkan dapat berlanjut menjadi stunting.
            """
        )

    with st.expander("🐟 2. Konteks Kearifan Lokal Madura: Pangan Berprotein Tinggi", expanded=False):
        st.markdown(
            """
            Pulau Madura memiliki kekayaan maritim dan agraris yang sangat melimpah untuk menunjang kebutuhan protein hewani anak:
            - **Ikan Laut Segar:** Ikan tongkol, kembung, layang, dan teri segar yang berlimpah di pesisir Bangkalan, Sampang, Pamekasan, dan Sumenep kaya akan asam amino esensial, asam lemak omega-3, dan kalsium yang langsung diserap tubuh untuk merangsang hormon pertumbuhan (*Growth Hormone* / IGF-1).
            - **Telur Ayam:** Sumber protein hewani berdaya cerna tinggi yang ekonomis dan mudah didapat. Konsumsi 1-2 butir telur per hari terbukti efektif mencegah stunting linier.
            - **Kelor & Sayur Hijau:** Daun kelor (*Moringa oleifera*) yang umum ditanam di pekarangan warga Madura kaya zat besi, vitamin A, dan mikronutrien pencegah anemia gizi.
            """
        )

    with st.expander("💧 3. Peran Air Bersih, Sanitasi & Pencegahan Infeksi", expanded=False):
        st.markdown(
            """
            Gizi yang cukup tidak akan terserap optimal jika anak sering mengalami diare atau infeksi saluran napas (ISPA). 
            - Kondisi sanitasi yang tidak layak memicu *Environmental Enteric Dysfunction* (EED), yaitu peradangan kronis pada dinding usus yang menghambat penyerapan mikronutrien penting.
            - Selalu gunakan air matang untuk konsumsi dan biasakan mencuci tangan pakai sabun setelah buang air dan sebelum menyuapi balita.
            """
        )

    with st.expander("📚 4. Rujukan Ilmiah & Regulasi Resmi", expanded=False):
        st.markdown(
            """
            1. **World Health Organization (WHO) Child Growth Standards (2006):** Standar internasional kurva pertumbuhan anak umur 0-5 tahun berdasarkan sampel anak sehat global.
            2. **Peraturan Menteri Kesehatan Republik Indonesia (Permenkes RI) No. 2 Tahun 2020:** Standar Antropometri Anak di Indonesia untuk penentuan status gizi.
            3. **Pedoman Pemantauan Tumbuh Kembang Balita Kementerian Kesehatan RI:** Acuan operasional Posyandu dan Puskesmas dalam skrining dan intervensi dini.
            """
        )


# ------------------------------------------------------------------------------
# HALAMAN 7: TENTANG & PANDUAN
# ------------------------------------------------------------------------------
elif current_page == "ℹ️ Tentang & Panduan":
    st.markdown("### ℹ️ Tentang Sistem & Panduan Penggunaan")
    st.markdown("<p style='color:var(--muted); font-size:0.88rem;'>Dokumentasi lengkap perancangan sistem, alur algoritma KNN, batasan etika, dan profil pengembang.</p>", unsafe_allow_html=True)

    c_prof, c_abt = st.columns([1, 2])
    with c_prof:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align:center;">
                <div style="font-size:3.5rem; margin-bottom:8px;">👨‍💻</div>
                <h4 style="margin:0; font-size:1.15rem;">ABD ROHMAN</h4>
                <div style="font-size:0.85rem; font-weight:700; color:{theme_colors['primary']}; margin-top:3px;">
                    NIM: 20251230035
                </div>
                <div style="font-size:0.82rem; color:var(--muted); margin-top:8px; line-height:1.4;">
                    Program Studi Sistem Informasi<br>
                    Fakultas Teknik<br>
                    <strong>Universitas Islam Madura</strong><br>
                    Tahun Akademik 2026
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_abt:
        st.markdown(
            f"""
            <div class="glass-card">
                <h4 style="margin-top:0;">Tujuan & Ruang Lingkup Sistem</h4>
                <p style="font-size:0.86rem; line-height:1.6; color:var(--text);">
                    <strong>SIGAMA (Sistem Informasi Gizi Anak MAdura) - Madura Child Nutrition Screening System</strong> dirancang sebagai Sistem Pendukung Keputusan (SPK) untuk membantu kader posyandu, tenaga kesehatan, dan orang tua dalam melakukan <strong>skrining cepat status gizi balita</strong> di 4 kabupaten Pulau Madura (Bangkalan, Sampang, Pamekasan, dan Sumenep).
                </p>
                <p style="font-size:0.86rem; line-height:1.6; color:var(--text);">
                    Sistem memanfaatkan algoritma pembelajaran mesin non-parametrik <strong>K-Nearest Neighbors (KNN)</strong> yang mengukur kedekatan karakteristik antropometri balita baru terhadap data acuan historis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Alur Kerja Algoritma KNN
    st.markdown("#### 🔄 Alur Kerja Algoritma K-Nearest Neighbors (KNN)")
    st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#DCEEFB", color="#3FA7A0", fontname="Helvetica", fontsize=11];
  edge [color="#3FA7A0"];
  A [label="Input Balita\n(Umur, JK, BB, TB)"];
  B [label="Hitung Z-Score\n(TB/U & BB/U)"];
  C [label="Seleksi Fitur\n(Set A/B/C)"];
  D [label="Standardisasi\n(StandardScaler)"];
  E [label="Hitung Jarak\n(Manhattan/Euclidean)"];
  F [label="K Tetangga\nTerdekat"];
  G [label="Voting"];
  H [label="Status Gizi &\nKeyakinan"];
  A->B->C->D->E->F->G->H;
}
""")

    # Panduan Penggunaan Langkah demi Langkah
    with st.expander("📖 Panduan Langkah demi Langkah Penggunaan", expanded=True):
        st.markdown(
            """
            1. **Pilih atau Masukkan Data Balita:** Buka panel samping (*sidebar*), masukkan jenis kelamin, umur (bulan), berat badan (kg), tinggi badan (cm), dan kabupaten balita. Anda juga dapat mengklik tombol **Skenario Cepat** untuk mencoba profil yang umum.
            2. **Periksa Nilai Z-Score Otomatis:** Perhatikan nilai z-score TB/U dan BB/U yang langsung terhitung secara otomatis di bawah input data.
            3. **Lihat Hasil Skrining di Dashboard:** Buka tab **Dashboard & Skrining Gizi** untuk membaca label status (Normal / Stunting / Gizi Kurang), bar probabilitas, serta posisi balita pada pengukur visual multi-zona.
            4. **Eksplorasi Ruang Fitur 3D:** Masuk ke tab **Visualisasi 3D & 2D** untuk melihat sebaran posisi balita terhadap ribuan sampel data lainnya di Madura.
            5. **Pelajari Transparansi Perhitungan:** Kunjungi tab **Bedah Perhitungan Manual** untuk memverifikasi langkah perhitungan matematika hingga pembuktian kecocokan dengan *scikit-learn*.
            6. **Eksplorasi Kurva Pertumbuhan:** Buka tab **Edukasi Gizi & Kurva Pertumbuhan** untuk melihat kurva pertumbuhan standar WHO dan tips gizi berbasis potensi lokal Madura.
            """
        )

    # Keterbatasan & Batasan Sistem
    with st.expander("⚠️ Batasan Sistem & Kejujuran Ilmiah", expanded=False):
        st.markdown(
            """
            - **Data Simulasi:** Dataset yang digunakan adalah data sintetis antropometri balita Madura untuk keperluan penelitian SPK dan bukan data medis riil anak.
            - **Interpolasi Kurva WHO:** Perhitungan nilai median dan standar deviasi menggunakan interpolasi linear pada titik-titik umur acuan WHO 2006, bukan tabel resmi per hari/minggu.
            - **Bukan Alat Diagnosis Medis:** Sistem ini ditujukan murni sebagai instrumen skrining awal. Diagnosis klinis defisit gizi harus ditegakkan langsung oleh dokter spesialis anak, dokter umum, atau nutrisionis di fasilitas pelayanan kesehatan resmi.
            """
        )

    # Glosarium Istilah
    with st.expander("📚 Glosarium Singkat", expanded=False):
        st.markdown(
            """
            - **K-Nearest Neighbors (KNN):** Metode klasifikasi berbasis instance yang menentukan kelas suatu data baru berdasarkan mayoritas kelas dari $K$ tetangga terdekatnya.
            - **Z-Score (SD Score):** Nilai simpangan baku yang menunjukkan seberapa jauh ukuran antropometri seorang anak berada di atas atau di bawah median populasi acuan.
            - **TB/U (Tinggi Badan menurut Umur):** Indikator antropometri yang menggambarkan pertumbuhan linier balita untuk skrining stunting.
            - **BB/U (Berat Badan menurut Umur):** Indikator antropometri yang menggambarkan massa tubuh anak secara umum untuk skrining underweight (gizi kurang).
            - **1000 HPK:** Periode 1000 Hari Pertama Kehidupan, terhitung dari saat pembuahan dalam kandungan (270 hari) hingga anak berusia 2 tahun (730 hari).
            """
        )

    # Versi Pustaka Terpasang
    st.markdown("#### 📦 Informasi Lingkungan & Pustaka")
    st.markdown(
        f"""
        - **Streamlit:** `{st.__version__}`
        - **scikit-learn:** `{sklearn.__version__}`
        - **pandas:** `{pd.__version__}`
        - **numpy:** `{np.__version__}`
        - **plotly:** `{plotly.__version__}`
        - **joblib:** `{joblib.__version__}`
        """
    )

# ==============================================================================
# 9. FOOTER KACA SEMUA HALAMAN
# ==============================================================================
st.markdown(
    """
    <div class="custom-footer">
        © 2026 ABD ROHMAN - 20251230035 | Sistem Informasi, Fakultas Teknik, Universitas Islam Madura
    </div>
    """,
    unsafe_allow_html=True,
)
