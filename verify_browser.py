"""
verify_browser.py
Verifikasi Visual dengan Streamlit Headless + Playwright (Langkah 9 C)
Smart Nutrition Status Classification System for Madura
Pembuat : ABD ROHMAN - 20251230035
Instansi: Sistem Informasi, Fakultas Teknik, Universitas Islam Madura 2026
"""

import subprocess
import time
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

def run_browser_verification():
    print("Memulai server Streamlit pada port 8501...")
    proc = subprocess.Popen(
        ["/home/eka/.local/share/pipx/venvs/streamlit/bin/streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    time.sleep(4) # Beri waktu server bangun

    out_dir = Path("screenshots")
    out_dir.mkdir(exist_ok=True)
    
    console_errors = []
    
    try:
        with sync_playwright() as p:
            print("Membuka browser Google Chrome headless...")
            browser = p.chromium.launch(executable_path="/usr/bin/google-chrome-stable", headless=True)
            
            # 1. Desktop 1440px Mode Siang
            context_desktop = browser.new_context(viewport={"width": 1440, "height": 900})
            page_d = context_desktop.new_page()
            page_d.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            
            print("Navigasi ke http://localhost:8501/?mode=siang...")
            page_d.goto("http://localhost:8501/?mode=siang", wait_until="networkidle")
            page_d.wait_for_selector(".banner-container", timeout=15000)
            page_d.wait_for_timeout(1000)
            
            # Cek horizontal scroll
            has_h_scroll_desktop = page_d.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth")
            print(f"Desktop 1440px - Ada horizontal scroll halaman: {has_h_scroll_desktop}")
            page_d.screenshot(path=str(out_dir / "desktop_1440px_day_dashboard.png"), full_page=False)
            print("Screenshot disimpan: desktop_1440px_day_dashboard.png")

            # 2. Desktop 1440px Mode Malam
            page_d.goto("http://localhost:8501/?mode=malam", wait_until="networkidle")
            page_d.wait_for_selector(".banner-container", timeout=15000)
            page_d.wait_for_timeout(1000)
            page_d.screenshot(path=str(out_dir / "desktop_1440px_night_dashboard.png"), full_page=False)
            print("Screenshot disimpan: desktop_1440px_night_dashboard.png")

            # 3. Mobile 375px Mode Siang (iPhone SE / standar smartphone)
            context_mobile = browser.new_context(viewport={"width": 375, "height": 812})
            page_m = context_mobile.new_page()
            page_m.goto("http://localhost:8501/?mode=siang", wait_until="networkidle")
            page_m.wait_for_selector(".banner-container", timeout=15000)
            page_m.wait_for_timeout(1000)
            has_h_scroll_mobile = page_m.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth")
            print(f"Mobile 375px - Ada horizontal scroll halaman: {has_h_scroll_mobile}")
            page_m.screenshot(path=str(out_dir / "mobile_375px_day_dashboard.png"), full_page=False)
            print("Screenshot disimpan: mobile_375px_day_dashboard.png")

            # 4. Mobile 375px Mode Malam
            page_m.goto("http://localhost:8501/?mode=malam", wait_until="networkidle")
            page_m.wait_for_selector(".banner-container", timeout=15000)
            page_m.wait_for_timeout(1000)
            page_m.screenshot(path=str(out_dir / "mobile_375px_night_dashboard.png"), full_page=False)
            print("Screenshot disimpan: mobile_375px_night_dashboard.png")

            browser.close()
            print("\nSemua screenshot browser berhasil diambil!")
            print(f"Console errors detected: {len(console_errors)}")
    finally:
        print("Menghentikan server Streamlit...")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

if __name__ == "__main__":
    run_browser_verification()
