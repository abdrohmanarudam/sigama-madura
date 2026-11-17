"""
ui_components.py
Komponen Antarmuka Kaca (Glassmorphism), Tabel HTML Responsif,
Pengukur Z-Score, Kartu Hasil, dan Visualisasi Terpadu.
Smart Nutrition Status Classification System for Madura
Pembuat: ABD ROHMAN - 20251230035
Instansi: Sistem Informasi, Fakultas Teknik, Universitas Islam Madura 2026
"""

from __future__ import annotations
import html
from typing import Dict, Any, List, Optional
import pandas as pd
from styles import get_theme_colors

def render_banner(is_dark: bool) -> str:
    c = get_theme_colors(is_dark)
    return f"""
    <div class="banner-container">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
            <div style="flex: 1; min-width: 280px;">
                <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.4rem;">
                    <span style="
                        display: inline-flex;
                        align-items: center;
                        justify-content: center;
                        width: 32px;
                        height: 32px;
                        border-radius: 10px;
                        background: {c["primary"]}25;
                        border: 1px solid {c["primary"]}50;
                    ">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{c["primary"]}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
                        </svg>
                    </span>
                    <span style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: {c["primary"]};">
                        Sistem Pendukung Keputusan SPK
                    </span>
                </div>
                <h1 class="banner-title">SIGAMA - Madura Child Nutrition Screening System</h1>
                <div style="font-size: clamp(0.9rem, 1.5vw, 1.05rem); font-weight: 600; color: {c['primary']}; margin-top: 0.25rem;">
                    (Sistem Informasi Gizi Anak MAdura)
                </div>
                <p class="banner-subtitle" style="margin-top: 0.35rem;">
                    Sistem Informasi Gizi Anak Madura: Skrining Status Gizi Balita Berbasis K-Nearest Neighbors (KNN)
                </p>
                <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; margin-top: 0.6rem;">
                    <div class="banner-chip">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                            <circle cx="12" cy="7" r="4"/>
                        </svg>
                        <strong>ABD ROHMAN</strong> (NIM: 20251230035)
                    </div>
                    <div class="banner-chip">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M22 10v6M2 10l10-5 10 5-10 5z"/>
                            <path d="M6 12v5c3 3 9 3 12 0v-5"/>
                        </svg>
                        Sistem Informasi, Fakultas Teknik
                    </div>
                    <div class="banner-chip">
                        🏛️ Universitas Islam Madura 2026
                    </div>
                </div>
            </div>
            <!-- Elemen 3D Visual Kaca Melayang (SVG Orisinal) -->
            <div style="display: flex; align-items: center; justify-content: center; padding: 0.5rem;">
                <svg width="105" height="105" viewBox="0 0 120 120" style="filter: drop-shadow(0 12px 20px rgba(0,0,0,0.15));">
                    <defs>
                        <radialGradient id="sphere1" cx="35%" cy="35%" r="65%">
                            <stop offset="0%" stop-color="{c["primary"]}" stop-opacity="0.9" />
                            <stop offset="60%" stop-color="{c["accent"]}" stop-opacity="0.6" />
                            <stop offset="100%" stop-color="{c["bg_to"]}" stop-opacity="0.3" />
                        </radialGradient>
                        <radialGradient id="sphere2" cx="30%" cy="30%" r="70%">
                            <stop offset="0%" stop-color="{c["accent2"]}" stop-opacity="0.85" />
                            <stop offset="100%" stop-color="{c["primary"]}" stop-opacity="0.2" />
                        </radialGradient>
                    </defs>
                    <!-- Bola Kaca Utama -->
                    <circle cx="68" cy="58" r="44" fill="url(#sphere1)" stroke="rgba(255,255,255,0.7)" stroke-width="2"/>
                    <!-- Bola Kaca Satelit -->
                    <circle cx="34" cy="84" r="22" fill="url(#sphere2)" stroke="rgba(255,255,255,0.6)" stroke-width="1.5"/>
                    <circle cx="86" cy="26" r="14" fill="rgba(255,255,255,0.65)" stroke="rgba(255,255,255,0.9)" stroke-width="1"/>
                    <!-- Ikon Anak & Hati di Pusat -->
                    <g transform="translate(48, 38) scale(1.6)" stroke="#ffffff" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="7" r="4"/>
                        <path d="M6 21v-2a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v2"/>
                    </g>
                </svg>
            </div>
        </div>
    </div>
    """


def render_disclaimer(is_dark: bool) -> str:
    c = get_theme_colors(is_dark)
    return f"""
    <div style="
        background: {c["warning"]}18;
        border: 1px solid {c["warning"]}45;
        border-radius: 16px;
        padding: 0.9rem 1.2rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.85rem;
    ">
        <span style="font-size: 1.4rem;">⚠️</span>
        <div style="font-size: 0.84rem; line-height: 1.45; color: {c["text"]};">
            <strong>Catatan Etika & Batasan Sistem:</strong> Hasil klasifikasi bersifat perkiraan untuk <strong>skrining awal</strong> berbasis data simulasi (sintetis), BUKAN alat diagnosis medis. Konfirmasi pengukuran dan penanganan wajib dilakukan oleh tenaga kesehatan resmi di posyandu atau puskesmas.
        </div>
    </div>
    """


def render_zscore_gauge(label: str, value: float, is_dark: bool) -> str:
    """
    Render visual gauge horizontal untuk skala -4 SD sampai +3 SD
    dengan zona warna lembut:
    < -3  : Sangat Pendek / Sangat Kurang (merah muda lembut)
    -3..-2: Pendek / Kurang (koral/kuning lembut)
    -2..+2: Normal (mint lembut)
    > +2  : Tinggi / Lebih (biru/lavender lembut)
    """
    c = get_theme_colors(is_dark)
    min_val, max_val = -4.0, 3.0
    clamped_val = max(min_val, min(max_val, value))
    pct = ((clamped_val - min_val) / (max_val - min_val)) * 100.0

    # Tentukan badge teks
    if value < -3.0:
        status_text = "Sangat Rendah (< -3 SD)"
        val_color = c["danger"]
    elif value < -2.0:
        status_text = "Rendah (-3 s/d -2 SD)"
        val_color = c["warning"]
    elif value <= 2.0:
        status_text = "Normal (-2 s/d +2 SD)"
        val_color = c["success"]
    else:
        status_text = "Tinggi (> +2 SD)"
        val_color = c["accent"]

    return f"""
    <div style="
        background: {c["glass_bg"]};
        border: 1px solid {c["glass_border"]};
        border-radius: 18px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.04);
    ">
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.5rem;">
            <span style="font-weight: 600; font-size: 0.95rem; color: {c["text"]};">{label}</span>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.25rem; font-weight: 800; color: {val_color};">{value:+.2f} SD</span>
                <span style="font-size: 0.78rem; font-weight: 600; padding: 0.2rem 0.55rem; border-radius: 8px; background: {val_color}22; color: {val_color}; border: 1px solid {val_color}44;">
                    {status_text}
                </span>
            </div>
        </div>
        
        <!-- Bar Pengukur Multi-Zona -->
        <div style="position: relative; height: 16px; border-radius: 8px; background: rgba(0,0,0,0.06); overflow: hidden; display: flex;">
            <!-- Zona < -3 (14.3%) -->
            <div style="width: 14.3%; background: {c["danger"]}40;" title="Sangat Rendah (< -3 SD)"></div>
            <!-- Zona -3 s/d -2 (14.3%) -->
            <div style="width: 14.3%; background: {c["warning"]}45;" title="Rendah (-3 s/d -2 SD)"></div>
            <!-- Zona -2 s/d +2 (57.1%) -->
            <div style="width: 57.1%; background: {c["success"]}50;" title="Normal (-2 s/d +2 SD)"></div>
            <!-- Zona > +2 (14.3%) -->
            <div style="width: 14.3%; background: {c["accent"]}45;" title="Tinggi (> +2 SD)"></div>
        </div>

        <!-- Indikator Posisi Jarum -->
        <div style="position: relative; height: 18px; margin-top: 2px;">
            <div style="
                position: absolute;
                left: {pct}%;
                transform: translateX(-50%);
                top: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
            ">
                <div style="
                    width: 0;
                    height: 0;
                    border-left: 6px solid transparent;
                    border-right: 6px solid transparent;
                    border-bottom: 7px solid {val_color};
                "></div>
                <div style="width: 3px; height: 8px; background: {val_color}; border-radius: 2px;"></div>
            </div>
        </div>

        <!-- Label Skala Bawah -->
        <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: {c["text_muted"]}; margin-top: -4px;">
            <span>-4 SD</span>
            <span>-3 SD</span>
            <span style="font-weight: 600; color: {c["danger"]};">-2 SD (Ambang)</span>
            <span>0 (Median)</span>
            <span>+2 SD</span>
            <span>+3 SD</span>
        </div>
    </div>
    """


def render_prediction_card(
    status: str,
    confidence: float,
    probabilities: Dict[str, float],
    k_neighbors_info: Dict[str, Any],
    is_dark: bool,
) -> str:
    c = get_theme_colors(is_dark)
    color_map = {
        "Normal": c["success"],
        "Gizi Kurang": c["warning"],
        "Stunting": c["danger"],
    }
    icon_map = {
        "Normal": "👶🌱",
        "Gizi Kurang": "⚖️🥣",
        "Stunting": "📏⚠️",
    }
    st_color = color_map.get(status, c["primary"])
    st_icon = icon_map.get(status, "🩺")

    # Bar probabilitas 3 kelas
    bars_html = ""
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    for cls_name, prob in sorted_probs:
        cls_color = color_map.get(cls_name, c["primary"])
        pct_prob = prob * 100
        bars_html += f"""
        <div style="margin-bottom: 0.75rem;">
            <div style="display: flex; justify-content: space-between; font-size: 0.84rem; margin-bottom: 0.25rem;">
                <span style="font-weight: 500; color: {c["text"]};">{cls_name}</span>
                <span style="font-weight: 700; color: {cls_color};">{pct_prob:.1f}%</span>
            </div>
            <div style="height: 10px; border-radius: 5px; background: rgba(0,0,0,0.06); overflow: hidden;">
                <div style="width: {pct_prob}%; height: 100%; border-radius: 5px; background: linear-gradient(90deg, {cls_color}99, {cls_color}); transition: width 0.6s ease;"></div>
            </div>
        </div>
        """

    k_val = k_neighbors_info.get("k", 18)
    majority_count = k_neighbors_info.get("majority_count", int(confidence * k_val))

    return f"""
    <div class="glass-card" style="border-left: 6px solid {st_color};">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
            <div>
                <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em; color: {c["text_muted"]}; font-weight: 600;">
                    Hasil Klasifikasi K-Nearest Neighbors
                </div>
                <div style="display: flex; align-items: center; gap: 0.85rem; margin-top: 0.35rem;">
                    <span style="font-size: 2.3rem;">{st_icon}</span>
                    <div>
                        <div style="font-family: 'Poppins', sans-serif; font-size: 2.1rem; font-weight: 800; color: {st_color}; line-height: 1.1;">
                            {status}
                        </div>
                        <div style="font-size: 0.85rem; color: {c["text_muted"]}; margin-top: 0.2rem;">
                            Tingkat Keyakinan: <strong>{confidence * 100:.1f}%</strong>
                        </div>
                    </div>
                </div>
            </div>
            <div style="
                background: {st_color}18;
                border: 1px solid {st_color}45;
                border-radius: 16px;
                padding: 0.75rem 1.25rem;
                text-align: right;
            ">
                <div style="font-size: 0.75rem; text-transform: uppercase; color: {c["text_muted"]}; font-weight: 600;">Komposisi K-Tetangga</div>
                <div style="font-size: 1.35rem; font-weight: 800; color: {st_color};">
                    {majority_count} <span style="font-size: 0.85rem; font-weight: 500; color: {c["text"]};">dari {k_val} tetangga</span>
                </div>
                <div style="font-size: 0.75rem; color: {c["text_muted"]};">Mendukung kelas {status}</div>
            </div>
        </div>

        <div style="margin-top: 1.4rem; padding-top: 1.1rem; border-top: 1px solid {c["glass_border"]};">
            <div style="font-size: 0.85rem; font-weight: 600; color: {c["text"]}; margin-bottom: 0.7rem;">
                Peringkat Probabilitas Status Gizi:
            </div>
            {bars_html}
        </div>
    </div>
    """


def render_risk_factor_panel(row_data: Dict[str, Any], is_dark: bool) -> str:
    c = get_theme_colors(is_dark)
    factors = [
        {
            "name": "Pemberian ASI Eksklusif",
            "val": row_data.get("asi_eksklusif", "Ya"),
            "is_good": row_data.get("asi_eksklusif", "Ya") == "Ya",
            "advice": "ASI eksklusif 6 bulan memberikan antibodi & nutrisi esensial optimal bagi balita."
            if row_data.get("asi_eksklusif") == "Ya"
            else "Perlu perhatian: balita tanpa ASI eksklusif berisiko lebih tinggi mengalami gangguan imun dan cerna.",
        },
        {
            "name": "Sanitasi & Jamban Layak",
            "val": row_data.get("sanitasi_layak", "Ya"),
            "is_good": row_data.get("sanitasi_layak", "Ya") == "Ya",
            "advice": "Lingkungan sanitasi bersih menurunkan risiko paparan enteropati lingkungan & diare."
            if row_data.get("sanitasi_layak") == "Ya"
            else "Perlu perhatian: akses sanitasi buruk meningkatkan risiko infeksi berulang penyebab stunting.",
        },
        {
            "name": "Episode Infeksi (3 Bln)",
            "val": f"{row_data.get('episode_infeksi_3bln', 0)} kali",
            "is_good": row_data.get("episode_infeksi_3bln", 0) <= 1,
            "advice": "Frekuensi infeksi minimal, penyerapan nutrisi usus terjaga baik."
            if row_data.get("episode_infeksi_3bln", 0) <= 1
            else "Perlu perhatian: riwayat infeksi sering (>=2 kali) menguras cadangan nutrisi tubuh balita.",
        },
        {
            "name": "Pendapatan Keluarga",
            "val": f"Rp {row_data.get('pendapatan_keluarga_juta', 2.5):.1f} jt/bln",
            "is_good": row_data.get("pendapatan_keluarga_juta", 2.5) >= 2.0,
            "advice": "Daya beli cukup untuk pemenuhan variasi pangan bergizi seimbang (protein hewani)."
            if row_data.get("pendapatan_keluarga_juta", 2.5) >= 2.0
            else "Perlu perhatian: daya beli terbatas, utamakan pangan lokal berprotein tinggi (ikan, telur, tahu/tempe).",
        },
    ]

    cards_html = ""
    for f in factors:
        badge_color = c["success"] if f["is_good"] else c["warning"]
        badge_text = "Baik" if f["is_good"] else "Perlu Perhatian"
        cards_html += f"""
        <div style="
            background: {c["glass_bg"]};
            border: 1px solid {c["glass_border"]};
            border-radius: 16px;
            padding: 0.95rem 1.1rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                <span style="font-weight: 600; font-size: 0.88rem; color: {c["text"]};">{f["name"]}</span>
                <span style="font-size: 0.75rem; font-weight: 600; padding: 0.2rem 0.55rem; border-radius: 8px; background: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}44;">
                    {badge_text}
                </span>
            </div>
            <div style="font-size: 1.05rem; font-weight: 700; color: {c["text"]}; margin-bottom: 0.35rem;">
                {f["val"]}
            </div>
            <div style="font-size: 0.77rem; color: {c["text_muted"]}; line-height: 1.4;">
                {f["advice"]}
            </div>
        </div>
        """

    return f"""
    <div class="glass-card">
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.3rem;">
            <h4 style="margin: 0; font-size: 1.05rem;">🛡️ Faktor Risiko & Determinan Tambahan</h4>
            <span style="font-size: 0.78rem; color: {c["text_muted"]};">Informasi Kontekstual Balita</span>
        </div>
        <p style="font-size: 0.82rem; color: {c["text_muted"]}; margin-bottom: 1rem;">
            <em>Catatan:</em> Bila model aktif menggunakan <strong>Set A (Z-Score Inti)</strong>, faktor risiko di bawah tidak memengaruhi perhitungan jarak KNN secara langsung, namun sangat penting sebagai evaluasi determinan kesehatan menyeluruh.
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 0.85rem;">
            {cards_html}
        </div>
    </div>
    """


def render_table(df: pd.DataFrame, page: int = 1, page_size: int = 50, is_dark: bool = False) -> str:
    """
    Helper render tabel ber-CSS kaca dengan header sticky, zebra halus,
    responsif scroll horizontal di mobile, dan pagination label.
    """
    c = get_theme_colors(is_dark)
    total_rows = len(df)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    curr_page = max(1, min(page, total_pages))
    start_idx = (curr_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)

    sub_df = df.iloc[start_idx:end_idx]

    # Generate header
    headers_html = "".join(
        f'<th style="padding: 10px 14px; text-align: left; font-weight: 600; font-size: 0.82rem; color: {c["text"]}; background: {c["glass_hover"]}; position: sticky; top: 0; border-bottom: 2px solid {c["glass_border"]}; white-space: nowrap;">{html.escape(str(col))}</th>'
        for col in sub_df.columns
    )

    # Generate rows
    rows_html = ""
    for r_idx, (_, row) in enumerate(sub_df.iterrows()):
        zebra_bg = "transparent" if r_idx % 2 == 0 else f"{c['glass_bg']}"
        cols_html = ""
        for val in row:
            val_str = f"{val:.2f}" if isinstance(val, (float, int)) and isinstance(val, float) else str(val)
            # Berikan badge pada kolom status_gizi jika ada
            if val_str == "Normal":
                cell_content = f'<span style="padding:2px 8px; border-radius:6px; background:{c["success"]}25; color:{c["success"]}; font-weight:600; font-size:0.8rem;">Normal</span>'
            elif val_str == "Stunting":
                cell_content = f'<span style="padding:2px 8px; border-radius:6px; background:{c["danger"]}25; color:{c["danger"]}; font-weight:600; font-size:0.8rem;">Stunting</span>'
            elif val_str == "Gizi Kurang":
                cell_content = f'<span style="padding:2px 8px; border-radius:6px; background:{c["warning"]}25; color:{c["warning"]}; font-weight:600; font-size:0.8rem;">Gizi Kurang</span>'
            else:
                cell_content = html.escape(val_str)
            cols_html += f'<td style="padding: 8px 14px; font-size: 0.83rem; color: {c["text"]}; border-bottom: 1px solid {c["glass_border"]}; white-space: nowrap;">{cell_content}</td>'

        rows_html += f'<tr style="background: {zebra_bg}; transition: background 150ms;">{cols_html}</tr>'

    return f"""
    <div style="margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; font-size: 0.82rem; color: {c["text_muted"]};">
            <span>Menampilkan data ke-<strong>{start_idx + 1}</strong> s/d <strong>{end_idx}</strong> dari <strong>{total_rows}</strong> total baris</span>
            <span>Halaman <strong>{curr_page}</strong> dari <strong>{total_pages}</strong></span>
        </div>
        <div style="
            background: {c["glass_bg"]};
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid {c["glass_border"]};
            border-radius: 18px;
            overflow: auto;
            max-height: 480px;
            box-shadow: {c["card_shadow"]};
        ">
            <table style="width: 100%; border-collapse: collapse; text-align: left;">
                <thead><tr>{headers_html}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
    </div>
    """
