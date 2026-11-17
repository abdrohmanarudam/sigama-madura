"""
styles.py
Manajemen Gaya Glassmorphism Lembut & 3D, Tema Cerah Adem (Siang & Malam),
Jam Digital Real-time, dan Loading Overlay.
Smart Nutrition Status Classification System for Madura
Pembuat: ABD ROHMAN - 20251230035
Instansi: Sistem Informasi, Fakultas Teknik, Universitas Islam Madura 2026
"""

from __future__ import annotations
from typing import Dict, Any, List

def hex_to_rgb(hex_str: str) -> tuple[float, float, float]:
    hex_clean = hex_str.lstrip("#")
    return tuple(int(hex_clean[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def rel_luminance(rgb: tuple[float, float, float]) -> float:
    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = [channel(c) for c in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def get_contrast_ratio(hex1: str, hex2: str) -> float:
    l1 = rel_luminance(hex_to_rgb(hex1))
    l2 = rel_luminance(hex_to_rgb(hex2))
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def get_theme_colors(is_dark: bool) -> Dict[str, str]:
    if is_dark:
        return {
            "bg_from": "#0F2430",
            "bg_via": "#12303F",
            "bg_to": "#1B2447",
            "glass_bg": "rgba(255, 255, 255, 0.08)",
            "glass_border": "rgba(255, 255, 255, 0.18)",
            "glass_hover": "rgba(255, 255, 255, 0.14)",
            "text": "#E4F1F5",
            "text_muted": "#9DB8C4",
            "primary": "#5CCBC2",
            "accent": "#86B9F2",
            "accent2": "#B9AEF0",
            "success": "#62CDAE",
            "warning": "#F5C76A",
            "danger": "#F29A82",
            "card_shadow": "0 10px 30px rgba(0, 0, 0, 0.35), 0 2px 6px rgba(0, 0, 0, 0.2)",
            "plotly_paper": "rgba(15, 36, 48, 0.4)",
            "plotly_grid": "rgba(255, 255, 255, 0.1)",
        }
    else:
        return {
            "bg_from": "#E6F6F4",
            "bg_via": "#DCEEFB",
            "bg_to": "#EFE9FB",
            "glass_bg": "rgba(255, 255, 255, 0.42)",
            "glass_border": "rgba(255, 255, 255, 0.70)",
            "glass_hover": "rgba(255, 255, 255, 0.58)",
            "text": "#1F3B4D",
            "text_muted": "#3E5D70",  # Disesuaikan agar kontras >= 4.5:1 (WCAG AA)
            "primary": "#3FA7A0",
            "accent": "#6AA6E8",
            "accent2": "#A99BE6",
            "success": "#4FB89A",
            "warning": "#F2B84B",
            "danger": "#EE8A6E",
            "card_shadow": "0 10px 30px rgba(31, 80, 120, 0.12), 0 2px 6px rgba(31, 80, 120, 0.06)",
            "plotly_paper": "rgba(255, 255, 255, 0.4)",
            "plotly_grid": "rgba(31, 59, 77, 0.08)",
        }


def check_wcag_contrast(is_dark: bool) -> List[Dict[str, Any]]:
    """
    Memeriksa rasio kontras teks utama terhadap latarnya berdasarkan pedoman WCAG AA (>= 4.5:1).
    """
    if is_dark:
        pairs = [
            ("Teks Utama pada Latar Halaman", "#E4F1F5", "#0F2430"),
            ("Teks Muted pada Latar Halaman", "#9DB8C4", "#0F2430"),
            ("Teks Utama pada Kartu Kaca", "#E4F1F5", "#162F3D"),
            ("Teks Muted pada Kartu Kaca", "#9DB8C4", "#162F3D"),
            ("Teks Aksen Primer", "#5CCBC2", "#0F2430"),
            ("Teks Badge Normal", "#62CDAE", "#0F2430"),
            ("Teks Badge Gizi Kurang", "#F5C76A", "#0F2430"),
            ("Teks Badge Stunting", "#F29A82", "#0F2430"),
        ]
    else:
        pairs = [
            ("Teks Utama pada Latar Halaman", "#1F3B4D", "#E6F6F4"),
            ("Teks Muted pada Latar Halaman", "#3E5D70", "#E6F6F4"),
            ("Teks Utama pada Kartu Kaca", "#1F3B4D", "#F4FAF9"),
            ("Teks Muted pada Kartu Kaca", "#3E5D70", "#F4FAF9"),
            ("Teks Tombol Primer (Putih)", "#FFFFFF", "#267B75"),
            ("Teks Badge Normal (Gelap)", "#145A48", "#E6F7F2"),
            ("Teks Badge Gizi Kurang (Gelap)", "#734E00", "#FDF5E6"),
            ("Teks Badge Stunting (Gelap)", "#842614", "#FDEFEA"),
        ]

    results = []
    for label, fg, background in pairs:
        ratio = get_contrast_ratio(fg, background)
        results.append(
            {
                "label": label,
                "fg": fg,
                "bg": background,
                "ratio": round(ratio, 2),
                "is_wcag_aa": ratio >= 4.5,
            }
        )
    return results


def get_width_kwargs(target: str = "button", stretch: bool = True) -> Dict[str, Any]:
    """
    Helper kompatibilitas Streamlit:
    Menyesuaikan antara 'use_container_width' dan 'width' tergantung versi Streamlit.
    """
    import inspect
    import streamlit as st_mod

    fn = getattr(st_mod, target, None)
    if fn is not None:
        try:
            sig = inspect.signature(fn)
            if "width" in sig.parameters:
                return {"width": "stretch" if stretch else "content"}
        except Exception:
            pass
    return {"use_container_width": stretch}


def get_theme_css(is_dark: bool) -> str:
    c = get_theme_colors(is_dark)
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@500;600;700;800&display=swap');

:root {{
    --bg-from: {c["bg_from"]};
    --bg-via: {c["bg_via"]};
    --bg-to: {c["bg_to"]};
    --glass: {c["glass_bg"]};
    --glass-border: {c["glass_border"]};
    --glass-hover: {c["glass_hover"]};
    --text: {c["text"]};
    --muted: {c["text_muted"]};
    --primary: {c["primary"]};
    --accent: {c["accent"]};
    --accent2: {c["accent2"]};
    --success: {c["success"]};
    --warning: {c["warning"]};
    --danger: {c["danger"]};
    --card-shadow: {c["card_shadow"]};
}}

/* Reset latar utama Streamlit */
.stApp {{
    background: linear-gradient(135deg, var(--bg-from) 0%, var(--bg-via) 50%, var(--bg-to) 100%) !important;
    background-attachment: fixed !important;
    color: var(--text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    min-height: 100vh;
}}

/* Latar mesh bergerak halus */
.stApp::before {{
    content: "";
    position: fixed;
    top: -10%;
    left: -10%;
    width: 120%;
    height: 120%;
    background: 
        radial-gradient(circle at 20% 30%, {c["primary"]}22 0%, transparent 40%),
        radial-gradient(circle at 80% 20%, {c["accent"]}25 0%, transparent 45%),
        radial-gradient(circle at 50% 80%, {c["accent2"]}20 0%, transparent 40%),
        radial-gradient(circle at 85% 75%, {c["success"]}18 0%, transparent 35%);
    filter: blur(60px);
    z-index: 0;
    pointer-events: none;
    animation: floatingMesh 24s ease-in-out infinite alternate;
}}

@keyframes floatingMesh {{
    0% {{ transform: translate(0, 0) scale(1); }}
    50% {{ transform: translate(25px, -30px) scale(1.04); }}
    100% {{ transform: translate(-20px, 20px) scale(0.98); }}
}}

@media (prefers-reduced-motion: reduce) {{
    .stApp::before {{
        animation: none !important;
    }}
}}

/* Konten di atas mesh */
.main .block-container {{
    position: relative;
    z-index: 1;
    max-width: 1200px;
    padding-top: 2rem !important;
    padding-bottom: 3.5rem !important;
}}

/* Typografi Utama */
h1, h2, h3, h4, h5, h6 {{
    font-family: 'Poppins', sans-serif !important;
    color: var(--text) !important;
    font-weight: 600;
}}

p, span, label, div {{
    color: var(--text);
}}

/* Sembunyikan footer Streamlit default & header bawaan */
footer {{
    display: none !important;
}}
header[data-testid="stHeader"] {{
    background: transparent !important;
    z-index: 99 !important;
}}

/* Kartu Kaca (Glassmorphism Card) */
.glass-card {{
    background: var(--glass);
    backdrop-filter: blur(18px) saturate(140%);
    -webkit-backdrop-filter: blur(18px) saturate(140%);
    border: 1px solid var(--glass-border);
    border-radius: 24px;
    padding: 1.5rem 1.8rem;
    box-shadow: var(--card-shadow);
    transition: transform 250ms ease, box-shadow 250ms ease, border-color 250ms ease;
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
}}

.glass-card::after {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.45), transparent);
    pointer-events: none;
}}

.glass-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.15), 0 4px 10px rgba(0, 0, 0, 0.08);
    border-color: rgba(255, 255, 255, 0.8);
}}

/* Banner Header */
.banner-container {{
    background: var(--glass);
    backdrop-filter: blur(20px) saturate(140%);
    -webkit-backdrop-filter: blur(20px) saturate(140%);
    border: 1px solid var(--glass-border);
    border-radius: 28px;
    padding: 1.8rem 2.2rem;
    box-shadow: var(--card-shadow);
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}}

.banner-title {{
    font-family: 'Poppins', sans-serif;
    font-size: clamp(1.4rem, 2.8vw, 2.1rem);
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.2;
    background: linear-gradient(135deg, var(--text) 30%, var(--primary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.banner-subtitle {{
    font-size: clamp(0.85rem, 1.4vw, 1.0rem);
    color: var(--muted);
    margin-top: 0.35rem;
    margin-bottom: 0.85rem;
    line-height: 1.4;
}}

.banner-chip {{
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.32rem 0.85rem;
    border-radius: 9999px;
    background: rgba(255, 255, 255, 0.35);
    border: 1px solid var(--glass-border);
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text);
    box-shadow: 0 2px 5px rgba(0,0,0,0.04);
}}

/* Sidebar Kaca */
[data-testid="stSidebar"] {{
    background: var(--glass) !important;
    backdrop-filter: blur(22px) saturate(140%) !important;
    -webkit-backdrop-filter: blur(22px) saturate(140%) !important;
    border-right: 1px solid var(--glass-border) !important;
}}

[data-testid="stSidebar"] .block-container {{
    padding-top: 1.5rem !important;
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
}}

/* Tombol Glassmorphism */
.stButton > button {{
    background: var(--glass) !important;
    backdrop-filter: blur(14px) !important;
    border: 1px solid var(--glass-border) !important;
    color: var(--text) !important;
    border-radius: 14px !important;
    padding: 0.55rem 1.1rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    min-height: 44px !important;
    transition: all 200ms ease !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06) !important;
}}

.stButton > button:hover {{
    transform: translateY(-2px) !important;
    background: var(--glass-hover) !important;
    border-color: var(--primary) !important;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12) !important;
    color: var(--primary) !important;
}}

.stButton > button:active {{
    transform: translateY(0px) !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1) !important;
}}

/* Navigasi Horizontal Glass Pill */
div[data-testid="stRadio"] > div[role="radiogroup"] {{
    display: flex !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    padding: 0.45rem !important;
    background: var(--glass) !important;
    backdrop-filter: blur(18px) saturate(140%) !important;
    -webkit-backdrop-filter: blur(18px) saturate(140%) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 20px !important;
    box-shadow: var(--card-shadow) !important;
    gap: 0.35rem !important;
    margin-bottom: 1.5rem !important;
    scrollbar-width: thin;
}}

div[data-testid="stRadio"] > div[role="radiogroup"] > label {{
    background: transparent !important;
    padding: 0.5rem 1.0rem !important;
    border-radius: 14px !important;
    border: 1px solid transparent !important;
    cursor: pointer !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: var(--text) !important;
    transition: all 200ms ease !important;
    white-space: nowrap !important;
}}

div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {{
    background: rgba(255, 255, 255, 0.25) !important;
    color: var(--primary) !important;
}}

div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] {{
    background: rgba(255, 255, 255, 0.55) !important;
    border-color: var(--primary) !important;
    color: var(--primary) !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
}}

/* Form Inputs, Selectbox, Slider Styling */
div[data-baseweb="select"] > div {{
    background: var(--glass) !important;
    border-color: var(--glass-border) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
}}

div[data-baseweb="popover"], ul[role="listbox"] {{
    background: {c["bg_via"]} !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.25) !important;
}}

ul[role="listbox"] li {{
    color: var(--text) !important;
}}

ul[role="listbox"] li:hover {{
    background: var(--primary)22 !important;
}}

/* Expander Kaca */
div[data-testid="stExpander"] {{
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 18px !important;
    margin-bottom: 0.85rem !important;
}}

/* Footer */
.custom-footer {{
    text-align: center;
    padding: 1.5rem 0.5rem 0.5rem 0.5rem;
    font-size: 0.82rem;
    color: var(--muted);
    border-top: 1px solid var(--glass-border);
    margin-top: 2.5rem;
}}

/* Responsif Mobile & Tablet */
@media (max-width: 768px) {{
    .banner-container {{
        padding: 1.2rem 1.4rem !important;
        border-radius: 20px !important;
    }}
    .glass-card {{
        padding: 1.2rem 1.2rem !important;
        border-radius: 18px !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] {{
        padding: 0.25rem !important;
        border-radius: 16px !important;
    }}
}}
</style>
"""


def get_realtime_clock_html(is_dark: bool) -> str:
    c = get_theme_colors(is_dark)
    return f"""
    <div id="realtime-clock-container" style="
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 9999px;
        background: {c["glass_bg"]};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {c["glass_border"]};
        color: {c["text"]};
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 13px;
        font-weight: 500;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    ">
        <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:{c["primary"]}; animation:pulseDot 2s infinite;"></span>
        <span id="realtime-clock-text">Memuat waktu...</span>
    </div>
    <style>
        @keyframes pulseDot {{
            0% {{ transform: scale(0.9); opacity: 0.6; }}
            50% {{ transform: scale(1.3); opacity: 1; }}
            100% {{ transform: scale(0.9); opacity: 0.6; }}
        }}
    </style>
    <script>
        function updateClock() {{
            try {{
                var now = new Date();
                var options = {{
                    timeZone: 'Asia/Jakarta',
                    weekday: 'long',
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                    hour12: false
                }};
                var str = now.toLocaleString('id-ID', options);
                var el = document.getElementById('realtime-clock-text');
                if (el) {{
                    el.innerText = str.replace(' pukul ', ' • ') + ' WIB';
                }}
            }} catch(e) {{
                var el = document.getElementById('realtime-clock-text');
                if (el) {{ el.innerText = new Date().toUTCString(); }}
            }}
        }}
        updateClock();
        setInterval(updateClock, 1000);
    </script>
    """


def get_loading_overlay_html(label: str, is_dark: bool) -> str:
    c = get_theme_colors(is_dark)
    backdrop_bg = "rgba(15, 23, 42, 0.78)" if is_dark else "rgba(240, 249, 255, 0.84)"
    modal_bg = "rgba(30, 41, 59, 0.95)" if is_dark else "rgba(255, 255, 255, 0.96)"
    card_border = "rgba(255, 255, 255, 0.22)" if is_dark else "rgba(14, 165, 233, 0.32)"
    ring_track = "rgba(255, 255, 255, 0.15)" if is_dark else "rgba(14, 165, 233, 0.18)"

    html_content = f"""<div id="sigama-modal-overlay" style="position:fixed !important;top:0 !important;left:0 !important;right:0 !important;bottom:0 !important;width:100vw !important;height:100vh !important;background:{backdrop_bg} !important;backdrop-filter:blur(18px) saturate(160%) !important;-webkit-backdrop-filter:blur(18px) saturate(160%) !important;z-index:2147483647 !important;display:flex !important;align-items:center !important;justify-content:center !important;pointer-events:all !important;overflow:hidden !important;margin:0 !important;padding:0 !important;cursor:wait !important;">
<div style="max-width:460px;width:88%;margin:auto;padding:2.6rem 2.2rem;background:{modal_bg};border:1.5px solid {card_border};border-radius:28px;box-shadow:0 30px 70px -10px rgba(0,0,0,0.5);text-align:center;display:flex;flex-direction:column;align-items:center;justify-content:center;animation:sigamaModalPop 0.22s cubic-bezier(0.16,1,0.3,1);">
<div style="position:relative;width:68px;height:68px;margin-bottom:1.4rem;">
<div style="position:absolute;top:0;left:0;width:100%;height:100%;border:5px solid {ring_track};border-top:5px solid {c['primary']};border-radius:50%;animation:spinDual 0.85s linear infinite;"></div>
<div style="position:absolute;top:9px;left:9px;width:48px;height:48px;border:4px solid transparent;border-bottom:4px solid {c['accent']};border-radius:50%;animation:spinDualRev 1.15s linear infinite;"></div>
</div>
<div style="font-family:'Poppins',sans-serif;font-size:1.22rem;font-weight:700;color:{c['text']};margin-bottom:0.45rem;letter-spacing:-0.01em;">{label}</div>
<div style="font-size:0.86rem;color:{c['text_muted']};font-weight:500;line-height:1.45;">Memproses komputasi antropometri & pembaruan data...</div>
<div style="margin-top:1.2rem;display:inline-flex;align-items:center;gap:7px;padding:6px 16px;background:{c['primary']}15;border:1px solid {c['primary']}30;border-radius:14px;font-size:0.78rem;font-weight:600;color:{c['primary']};">
<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{c['primary']};animation:sigamaPulse 1s ease-in-out infinite;"></span>
Memuat... Mohon tunggu sejenak
</div>
<style>
@keyframes spinDual {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
@keyframes spinDualRev {{ 0% {{ transform: rotate(360deg); }} 100% {{ transform: rotate(0deg); }} }}
@keyframes sigamaModalPop {{ 0% {{ opacity: 0; transform: scale(0.92); }} 100% {{ opacity: 1; transform: scale(1); }} }}
@keyframes sigamaPulse {{ 0%, 100% {{ opacity: 0.35; transform: scale(0.85); }} 50% {{ opacity: 1; transform: scale(1.25); }} }}
</style>
</div>
</div>"""
    return html_content
