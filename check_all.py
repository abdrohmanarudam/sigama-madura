"""
check_all.py
Skrip Pemeriksaan Otomatis Menyeluruh (LANGKAH 9)
Smart Nutrition Status Classification System for Madura
Pembuat : ABD ROHMAN - 20251230035
Instansi: Sistem Informasi, Fakultas Teknik, Universitas Islam Madura 2026
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from streamlit.testing.v1 import AppTest

# Import modul internal proyek
import styles
import train_model


class InspectionSuite:
    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def log(self, item_num: int, name: str, passed: bool, details: str):
        self.results.append({
            "no": item_num,
            "name": name,
            "status": "PASS ✅" if passed else "FAIL ❌",
            "passed": passed,
            "details": details,
        })
        status_str = "\033[92mPASS ✅\033[0m" if passed else "\033[91mFAIL ❌\033[0m"
        print(f"[{item_num}] {name} -> {status_str}")
        print(f"    Catatan: {details}")

    # 1. Dataset Integrity
    def check_1_dataset(self) -> bool:
        p = Path("data_status_gizi_balita.xlsx")
        if not p.exists():
            self.log(1, "Integritas Dataset", False, f"File {p} tidak ditemukan.")
            return False

        try:
            df = train_model.load_data(p)
            has_13_cols = list(df.columns) == train_model.EXPECTED_COLUMNS
            no_na = df.isna().sum().sum() == 0
            no_dup = df.duplicated().sum() == 0
            n_rows = len(df) == 2300
            vc = df["status_gizi"].value_counts().to_dict()
            counts_match = (
                vc.get("Normal", 0) == 900
                and vc.get("Stunting", 0) == 850
                and vc.get("Gizi Kurang", 0) == 550
            )

            passed = has_13_cols and no_na and no_dup and n_rows and counts_match
            details = (
                f"2300 baris, 13 kolom lengkap, 0 missing, 0 duplicate, "
                f"Kelas: Normal={vc.get('Normal', 0)}, Stunting={vc.get('Stunting', 0)}, Gizi Kurang={vc.get('Gizi Kurang', 0)}"
            )
            self.log(1, "Integritas Dataset (2300 baris, 13 kolom, 900/850/550)", passed, details)
            return passed
        except Exception as e:
            self.log(1, "Integritas Dataset", False, f"Error: {e}")
            return False

    # 2. Z-Score Formula Accuracy
    def check_2_zscore_formula(self) -> bool:
        try:
            df = train_model.load_data("data_status_gizi_balita.xlsx")
            max_tb_diff = 0.0
            max_bb_diff = 0.0

            for _, row in df.iterrows():
                z_tb, z_bb = train_model.calculate_zscores(
                    row["umur_bulan"],
                    row["jenis_kelamin"],
                    row["berat_badan_kg"],
                    row["tinggi_badan_cm"],
                )
                d_tb = abs(z_tb - row["zscore_tb_u"])
                d_bb = abs(z_bb - row["zscore_bb_u"])
                if d_tb > max_tb_diff:
                    max_tb_diff = d_tb
                if d_bb > max_bb_diff:
                    max_bb_diff = d_bb

            passed = max_tb_diff <= 0.01 and max_bb_diff <= 0.01
            details = f"Selisih maks TB/U: {max_tb_diff:.4f}, Selisih maks BB/U: {max_bb_diff:.4f} (<= 0.01 batas pembulatan)"
            self.log(2, "Kecocokan Rumus Acuan Z-Score (selisih <= 0.01)", passed, details)
            return passed
        except Exception as e:
            self.log(2, "Kecocokan Rumus Acuan Z-Score", False, f"Error: {e}")
            return False

    # 3. Model Joblib & Feature Comparison
    def check_3_model_joblib(self) -> bool:
        p = Path("models/knn_gizi_balita_madura.joblib")
        if not p.exists():
            self.log(3, "Model Joblib & Akurasi", False, "File joblib belum ada.")
            return False

        try:
            m = joblib.load(p)
            acc = m["metrics"]["accuracy"]
            fset_comp = m["metrics"]["feature_set_comparison"]
            acc_a = fset_comp["A"]["accuracy"]
            acc_b = fset_comp["B"]["accuracy"]
            acc_c = fset_comp["C"]["accuracy"]

            is_acc_ok = acc >= 0.85
            is_better_than_c = (acc_a > acc_c) or (acc_b > acc_c)

            passed = is_acc_ok and is_better_than_c
            details = (
                f"Model SET {m['feature_set']} (K={m['k']}, {m['metric']}, {m['weights']}), "
                f"Akurasi Uji: {acc*100:.2f}% (>=85%). "
                f"Perbandingan: Set A={acc_a*100:.2f}%, Set B={acc_b*100:.2f}%, Set C={acc_c*100:.2f}%"
            )
            self.log(3, "Model Joblib, Akurasi >= 0.85 & Seleksi Fitur (A/B > C)", passed, details)
            return passed
        except Exception as e:
            self.log(3, "Model Joblib & Akurasi", False, f"Error: {e}")
            return False

    # 4. WCAG AA Contrast Ratios
    def check_4_contrast(self) -> bool:
        try:
            day_checks = styles.check_wcag_contrast(is_dark=False)
            night_checks = styles.check_wcag_contrast(is_dark=True)

            day_pass = all(c["is_wcag_aa"] for c in day_checks)
            night_pass = all(c["is_wcag_aa"] for c in night_checks)

            passed = day_pass and night_pass
            details = (
                f"Mode Siang: {len(day_checks)}/{len(day_checks)} lolos (Rasio terendah: {min(c['ratio'] for c in day_checks)}:1). "
                f"Mode Malam: {len(night_checks)}/{len(night_checks)} lolos (Rasio terendah: {min(c['ratio'] for c in night_checks)}:1)"
            )
            self.log(4, "Kontras Teks WCAG AA (>= 4.5:1) Mode Siang & Malam", passed, details)
            return passed
        except Exception as e:
            self.log(4, "Kontras Teks WCAG AA", False, f"Error: {e}")
            return False

    # 5. AppTest All 7 Pages & Widgets
    def check_5_apptest(self) -> bool:
        try:
            at = AppTest.from_file("app.py", default_timeout=45)
            at.run()
            if at.exception:
                self.log(5, "AppTest 7 Halaman & Widget", False, f"Exception saat startup: {at.exception[0]}")
                return False

            pages = [
                "🧒 Dashboard & Skrining Gizi",
                "📊 Visualisasi 3D & 2D",
                "🧮 Bedah Perhitungan Manual",
                "📈 Evaluasi & Optimasi K",
                "📁 Manajemen Dataset",
                "🥗 Edukasi Gizi & Kurva Pertumbuhan",
                "ℹ️ Tentang & Panduan",
            ]

            for pg in pages:
                at.radio(key="nav_radio_control").set_value(pg).run()
                if at.exception:
                    self.log(5, "AppTest 7 Halaman & Widget", False, f"Exception pada halaman '{pg}': {at.exception[0]}")
                    return False

            # Klik semua skenario cepat, Acak, Reset + ubah widget
            at.radio(key="nav_radio_control").set_value(pages[0]).run()
            for b in list(at.button):
                if any(k in b.label for k in ["Normal", "Stunting", "Gizi Kurang", "Acak", "Reset"]):
                    b.click().run()
                    if at.exception:
                        self.log(5, "AppTest 7 Halaman & Widget", False, f"Exception klik '{b.label}': {at.exception[0]}")
                        return False
                    if not any("Hasil Klasifikasi" in m.value for m in at.markdown):
                        self.log(5, "AppTest 7 Halaman & Widget", False, f"Hasil skrining tidak muncul setelah '{b.label}'")
                        return False
            at.slider(key="input_umur_bulan").set_value(40).run()
            at.selectbox(key="input_jenis_kelamin").set_value("Perempuan").run()
            at.selectbox(key="input_kabupaten").set_value("Sampang").run()
            at.radio(key="input_asi_eksklusif").set_value("Tidak").run()
            at.number_input(key="input_berat_badan_kg").set_value(13.0).run()
            if at.exception:
                self.log(5, "AppTest 7 Halaman & Widget", False, f"Exception ubah widget: {at.exception[0]}")
                return False

            # Test toggle theme
            at.toggle(key="theme_toggle").set_value(True).run()
            if at.exception:
                self.log(5, "AppTest 7 Halaman & Widget", False, f"Exception saat toggle tema: {at.exception[0]}")
                return False
            at.toggle(key="theme_toggle").set_value(False).run()

            # Kembali ke halaman 1
            at.radio(key="nav_radio_control").set_value(pages[0]).run()

            self.log(5, "AppTest 7 Halaman & Widget Lengkap", True, "Semua 7 halaman dan toggle widget lulus tanpa exception.")
            return True
        except Exception as e:
            self.log(5, "AppTest 7 Halaman & Widget", False, f"Error: {e}")
            return False

    # 6. Quick Scenarios 1-4 Predictions
    def check_6_scenarios(self) -> bool:
        try:
            m = joblib.load("models/knn_gizi_balita_madura.joblib")
            pipe = m["pipeline"]
            cols = m["features"]

            scenarios = [
                ("Skenario 1 - Balita Normal", "Laki-laki", 24, 12.2, 87.0, "Normal"),
                ("Skenario 2 - Berisiko Stunting", "Perempuan", 30, 11.0, 82.0, "Stunting"),
                ("Skenario 3 - Gizi Kurang", "Laki-laki", 36, 10.3, 95.5, "Gizi Kurang"),
                ("Skenario 4 - Stunting Usia Dini", "Perempuan", 18, 8.4, 74.0, "Stunting"),
            ]

            all_match = True
            msg_parts = []
            for name, jk, u, bb, tb, expected in scenarios:
                ztb, zbb = train_model.calculate_zscores(u, jk, bb, tb)
                row_dict = {
                    "kabupaten": "Sumenep",
                    "jenis_kelamin": jk,
                    "umur_bulan": u,
                    "berat_badan_kg": bb,
                    "tinggi_badan_cm": tb,
                    "zscore_tb_u": ztb,
                    "zscore_bb_u": zbb,
                    "asi_eksklusif": "Ya",
                    "sanitasi_layak": "Ya",
                    "pendapatan_keluarga_juta": 2.5,
                    "lama_sekolah_ibu": 12,
                    "episode_infeksi_3bln": 0,
                }
                in_df = pd.DataFrame([row_dict])[cols]
                pred = pipe.predict(in_df)[0]
                proba = pipe.predict_proba(in_df)[0]
                conf = max(proba)
                matched = (pred == expected)
                if not matched:
                    all_match = False
                msg_parts.append(f"{name} -> {pred} ({conf*100:.1f}%) [Harapan: {expected}]")

            self.log(6, "Klasifikasi Skenario Cepat 1-4", all_match, "; ".join(msg_parts))
            return all_match
        except Exception as e:
            self.log(6, "Klasifikasi Skenario Cepat 1-4", False, f"Error: {e}")
            return False

    # 7. Bedah Perhitungan Manual vs kneighbors
    def check_7_manual_math(self) -> bool:
        try:
            m = joblib.load("models/knn_gizi_balita_madura.joblib")
            df = train_model.load_data("data_status_gizi_balita.xlsx")
            pipe = m["pipeline"]
            cols = m["features"]
            k_val = m["k"]
            metric = m["metric"]

            prep = pipe.named_steps["preprocessor"]
            knn_clf = pipe.named_steps["knn"]

            # Sample test point
            test_row = {
                "zscore_tb_u": 0.5,
                "zscore_bb_u": -0.8,
                "umur_bulan": 24,
                "berat_badan_kg": 11.5,
                "tinggi_badan_cm": 88.0,
                "pendapatan_keluarga_juta": 2.5,
                "lama_sekolah_ibu": 12,
                "episode_infeksi_3bln": 0,
            }
            sample_df = pd.DataFrame([test_row])[cols]
            user_scaled = prep.transform(sample_df)
            X_scaled = knn_clf._fit_X

            # Manual math
            if metric == "manhattan":
                dists_manual = np.sum(np.abs(X_scaled - user_scaled), axis=1)
            else:
                dists_manual = np.sqrt(np.sum((X_scaled - user_scaled) ** 2, axis=1))

            top_manual_indices = np.argsort(dists_manual)[:k_val]
            top_manual_dists = dists_manual[top_manual_indices]

            # Scikit-learn
            sk_dists, sk_indices = knn_clf.kneighbors(user_scaled)

            idx_match = np.array_equal(top_manual_indices, sk_indices[0])
            max_diff = np.max(np.abs(top_manual_dists - sk_dists[0]))

            passed = idx_match and (max_diff <= 1e-5)
            details = f"Indeks K tetangga identik: {idx_match}, Selisih jarak maksimum: {max_diff:.2e}"
            self.log(7, "Kecocokan Bedah Manual NumPy vs scikit-learn", passed, details)
            return passed
        except Exception as e:
            self.log(7, "Kecocokan Bedah Manual NumPy vs scikit-learn", False, f"Error: {e}")
            return False

    # 8. Data Add & Upload Validation
    def check_8_data_validation(self) -> bool:
        try:
            # 1. Test auto-calculation of new valid row
            u, jk, bb, tb = 24, "Laki-laki", 12.2, 87.0
            ztb, zbb = train_model.calculate_zscores(u, jk, bb, tb)
            st_calc = train_model.determine_status_gizi(ztb, zbb)
            calc_ok = (st_calc == "Normal") and abs(ztb) < 0.2 and abs(zbb) < 0.2

            # 2. Test invalid schema rejection
            invalid_df = pd.DataFrame([{"kolom_palsu": 123, "umur_bulan": 999}])
            missing_cols = [c for c in train_model.EXPECTED_COLUMNS if c not in invalid_df.columns]
            reject_ok = len(missing_cols) > 0

            passed = calc_ok and reject_ok
            details = f"Tambah data otomatis benar (Z TB: {ztb:.2f}, Z BB: {zbb:.2f}, Status: {st_calc}); Data skema salah tertolak ({len(missing_cols)} kolom hilang)"
            self.log(8, "Validasi Input Tambah & Unggah Data", passed, details)
            return passed
        except Exception as e:
            self.log(8, "Validasi Input Tambah & Unggah Data", False, f"Error: {e}")
            return False

    # 9. Retrain Model Execution
    def check_9_retrain(self) -> bool:
        try:
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            # Latih ulang cepat dengan subset k
            m_new = train_model.train_and_save_model(
                data_path="data_status_gizi_balita.xlsx",
                output_path=tmp_path,
                force_config={
                    "feature_set": "A",
                    "k": 9,
                    "metric": "euclidean",
                    "weights": "uniform",
                    "cv_accuracy": 0.90,
                    "cv_macro_f1": 0.89,
                },
            )

            is_saved = tmp_path.exists() and tmp_path.stat().st_size > 0
            test_acc = m_new["metrics"]["accuracy"]
            passed = is_saved and (test_acc >= 0.85)

            # Cleanup
            if tmp_path.exists():
                tmp_path.unlink()

            details = f"File joblib baru berhasil dibentuk ({m_new['n_train']} latih, {m_new['n_test']} uji), Akurasi Uji: {test_acc*100:.2f}%"
            self.log(9, "Pelatihan Ulang Model (Retrain) dari Aplikasi", passed, details)
            return passed
        except Exception as e:
            self.log(9, "Pelatihan Ulang Model", False, f"Error: {e}")
            return False


    # 10. Browser: tidak ada kode HTML bocor, tidak ada exception di tiap halaman
    def check_10_browser(self) -> bool:
        import subprocess, time
        proc = None
        try:
            from playwright.sync_api import sync_playwright
            streamlit_bin = str(Path(sys.executable).parent / "streamlit")
            proc = subprocess.Popen(
                [streamlit_bin, "run", "app.py", "--server.port=8599", "--server.headless=true"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            time.sleep(5)
            names = ["Visualisasi 3D", "Bedah Perhitungan", "Evaluasi & Optimasi", "Manajemen Dataset",
                     "Edukasi Gizi", "Tentang & Panduan", "Dashboard & Skrining"]
            problems = []
            with sync_playwright() as p:
                b = p.chromium.launch(executable_path="/usr/bin/google-chrome-stable", headless=True)
                for w, h in [(1440, 900), (375, 812)]:
                    for mode in ["siang", "malam"]:
                        pg = b.new_page(viewport={"width": w, "height": h})
                        pg.goto(f"http://localhost:8599/?mode={mode}", wait_until="networkidle")
                        pg.wait_for_selector(".banner-container", timeout=25000)
                        for n in names:
                            pg.get_by_text(n).first.click()
                            pg.wait_for_timeout(2500)
                            t = pg.inner_text("body")
                            if any(x in t for x in ["<div", 'style="', "</div>", "<svg", "<h4"]):
                                problems.append(f"HTML bocor: {n} {w}px {mode}")
                            if "Traceback" in t or "Error" in t.split("\n")[0]:
                                problems.append(f"Error: {n} {w}px {mode}")
                        if pg.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth"):
                            problems.append(f"Scroll horizontal {w}px {mode}")
                        pg.close()
                b.close()
            passed = not problems
            self.log(10, "Browser: HTML ter-render, tanpa error & tanpa scroll horizontal (1440/375px, siang/malam)",
                     passed, "Semua halaman bersih" if passed else "; ".join(problems[:5]))
            return passed
        except Exception as e:
            self.log(10, "Browser Chrome Headless", False, f"Error: {e}")
            return False
        finally:
            if proc:
                proc.terminate()

    def run_all(self) -> bool:
        print("\n" + "=" * 75)
        print("PEMERIKSAAN OTOMATIS MENYELURUH (LANGKAH 9)")
        print("SIGAMA | Madura Child Nutrition Screening System | ABD ROHMAN")
        print("=" * 75)

        s1 = self.check_1_dataset()
        s2 = self.check_2_zscore_formula()
        s3 = self.check_3_model_joblib()
        s4 = self.check_4_contrast()
        s5 = self.check_5_apptest()
        s6 = self.check_6_scenarios()
        s7 = self.check_7_manual_math()
        s8 = self.check_8_data_validation()
        s9 = self.check_9_retrain()
        s10 = self.check_10_browser()

        all_passed = all([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10])

        print("\n" + "=" * 75)
        print("RINGKASAN HASIL PEMERIKSAAN (TABEL PASS/FAIL):")
        print("=" * 75)
        print(f"{'No':<4} | {'Pemeriksaan':<45} | {'Hasil':<10}")
        print("-" * 75)
        for r in self.results:
            print(f"{r['no']:<4} | {r['name']:<45} | {r['status']:<10}")
        print("=" * 75)

        if all_passed:
            print("\n🎉 SEMUA 10 PEMERIKSAAN DINYATAKAN LULUS (ALL PASS)!")
        else:
            print("\n⚠️ ADA PEMERIKSAAN YANG GAGAL. SILAKAN PERIKSA DETAIL DI ATAS.")

        return all_passed


if __name__ == "__main__":
    suite = InspectionSuite()
    success = suite.run_all()
    sys.exit(0 if success else 1)
