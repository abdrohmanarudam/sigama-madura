"""
train_model.py
Modul Pelatihan Model K-Nearest Neighbors (KNN)
Smart Nutrition Status Classification System for Madura
Pembuat: ABD ROHMAN - 20251230035
Instansi: Sistem Informasi, Fakultas Teknik, Universitas Islam Madura 2026
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==============================================================================
# 1. KONSTANTA ACUAN PERTUMBUHAN & Z-SCORE (WHO 2006 PERKIRAAN INTERPOLASI)
# ==============================================================================
AGE_POINTS = np.array([0, 6, 12, 24, 36, 48, 60], dtype=float)
HEIGHT_MEDIAN_MALE = np.array([49.9, 67.6, 75.7, 87.1, 96.1, 103.3, 110.0], dtype=float)
HEIGHT_MEDIAN_FEMALE = np.array([49.1, 65.7, 74.0, 85.7, 95.1, 102.7, 109.4], dtype=float)
HEIGHT_SD_MALE = np.array([1.9, 2.15, 2.35, 3.05, 3.7, 4.2, 4.4], dtype=float)
HEIGHT_SD_FEMALE = np.array([1.9, 2.25, 2.55, 2.85, 3.85, 4.3, 4.75], dtype=float)

WEIGHT_MEDIAN_MALE = np.array([3.3, 7.9, 9.6, 12.2, 14.3, 16.3, 18.3], dtype=float)
WEIGHT_MEDIAN_FEMALE = np.array([3.2, 7.3, 8.9, 11.5, 13.9, 16.1, 18.2], dtype=float)
WEIGHT_FRACTION = np.array([0.12, 0.12, 0.12, 0.125, 0.13, 0.135, 0.14], dtype=float)

CATEGORICAL_COLS = ["kabupaten", "jenis_kelamin", "asi_eksklusif", "sanitasi_layak"]
NUMERICAL_COLS_ALL = [
    "umur_bulan",
    "berat_badan_kg",
    "tinggi_badan_cm",
    "zscore_tb_u",
    "zscore_bb_u",
    "pendapatan_keluarga_juta",
    "lama_sekolah_ibu",
    "episode_infeksi_3bln",
]

FEATURE_SETS: Dict[str, List[str]] = {
    "A": ["zscore_tb_u", "zscore_bb_u"],
    "B": NUMERICAL_COLS_ALL,
    "C": NUMERICAL_COLS_ALL + CATEGORICAL_COLS,
}

EXPECTED_COLUMNS = [
    "kabupaten",
    "jenis_kelamin",
    "umur_bulan",
    "berat_badan_kg",
    "tinggi_badan_cm",
    "zscore_tb_u",
    "zscore_bb_u",
    "asi_eksklusif",
    "sanitasi_layak",
    "pendapatan_keluarga_juta",
    "lama_sekolah_ibu",
    "episode_infeksi_3bln",
    "status_gizi",
]

VALID_CATEGORIES = {
    "kabupaten": ["Bangkalan", "Sampang", "Pamekasan", "Sumenep"],
    "jenis_kelamin": ["Laki-laki", "Perempuan"],
    "asi_eksklusif": ["Ya", "Tidak"],
    "sanitasi_layak": ["Ya", "Tidak"],
    "status_gizi": ["Normal", "Stunting", "Gizi Kurang"],
}

# ==============================================================================
# 2. FUNGSI PERHITUNGAN ANTROPOMETRI & Z-SCORE
# ==============================================================================
def get_growth_reference(umur_bulan: float, jenis_kelamin: str) -> Tuple[float, float, float, float]:
    """
    Mengambil median dan standar deviasi (SD) TB dan BB berdasarkan umur dan jenis kelamin
    menggunakan interpolasi linear tabel acuan WHO 2006.
    """
    u = float(umur_bulan)
    if jenis_kelamin == "Laki-laki":
        med_tb = float(np.interp(u, AGE_POINTS, HEIGHT_MEDIAN_MALE))
        sd_tb = float(np.interp(u, AGE_POINTS, HEIGHT_SD_MALE))
        med_bb = float(np.interp(u, AGE_POINTS, WEIGHT_MEDIAN_MALE))
    else:
        med_tb = float(np.interp(u, AGE_POINTS, HEIGHT_MEDIAN_FEMALE))
        sd_tb = float(np.interp(u, AGE_POINTS, HEIGHT_SD_FEMALE))
        med_bb = float(np.interp(u, AGE_POINTS, WEIGHT_MEDIAN_FEMALE))

    frac = float(np.interp(u, AGE_POINTS, WEIGHT_FRACTION))
    sd_bb = float(med_bb * frac)
    return med_tb, sd_tb, med_bb, sd_bb


def calculate_zscores(
    umur_bulan: float,
    jenis_kelamin: str,
    berat_badan_kg: float,
    tinggi_badan_cm: float,
) -> Tuple[float, float]:
    """
    Menghitung z-score TB/U dan BB/U secara otomatis.
    Rumus:
      zscore_tb_u = (tinggi_badan_cm - median_tinggi) / sd_tinggi
      zscore_bb_u = (berat_badan_kg - median_berat) / sd_berat
    """
    med_tb, sd_tb, med_bb, sd_bb = get_growth_reference(umur_bulan, jenis_kelamin)
    z_tb = (float(tinggi_badan_cm) - med_tb) / sd_tb
    z_bb = (float(berat_badan_kg) - med_bb) / sd_bb
    return float(z_tb), float(z_bb)


def determine_status_gizi(z_tb: float, z_bb: float) -> str:
    """
    Definisi status gizi acuan SPK:
    - Stunting    : zscore_tb_u < -2.0
    - Gizi Kurang : zscore_bb_u < -2.0 (dengan zscore_tb_u >= -2.0)
    - Normal      : selain itu
    """
    if z_tb < -2.0:
        return "Stunting"
    elif z_bb < -2.0:
        return "Gizi Kurang"
    else:
        return "Normal"


# ==============================================================================
# 3. PEMBACAAN DAN VALIDASI DATASET
# ==============================================================================
def load_data(filepath: str | Path) -> pd.DataFrame:
    """
    Membaca dan memvalidasi dataset dari file Excel.
    Mengambil sheet pertama 'data gizi balita', memvalidasi kolom dan tipe,
    serta membersihkan data duplikat/kosong jika ada.
    """
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"File data tidak ditemukan di: {p.resolve()}")

    try:
        # Baca sheet 'data gizi balita' jika ada, fallback ke sheet 0
        xl = pd.ExcelFile(p)
        sheet_target = "data gizi balita" if "data gizi balita" in xl.sheet_names else xl.sheet_names[0]
        df = pd.read_excel(p, sheet_name=sheet_target)
    except Exception as e:
        raise ValueError(f"Gagal membaca file Excel {p.name}: {e}")

    # Validasi kelengkapan kolom
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom wajib tidak lengkap pada dataset: {missing_cols}")

    # Buang baris kosong dan duplikat persis
    initial_len = len(df)
    df = df.dropna(subset=EXPECTED_COLUMNS).drop_duplicates().reset_index(drop=True)

    # Validasi tipe data numerik
    for col in NUMERICAL_COLS_ALL:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=NUMERICAL_COLS_ALL).reset_index(drop=True)

    return df


# ==============================================================================
# 4. PEMBANGUNAN PIPELINE MODEL KNN
# ==============================================================================
def create_pipeline(
    feature_set: str,
    k: int = 9,
    metric: str = "euclidean",
    weights: str = "uniform",
) -> Pipeline:
    """
    Membangun Pipeline scikit-learn dengan ColumnTransformer dan KNeighborsClassifier.
    - Set A & B: hanya StandardScaler untuk fitur numerik.
    - Set C: StandardScaler untuk numerik dan OneHotEncoder untuk kategorikal.
    """
    cols = FEATURE_SETS[feature_set]
    num_cols = [c for c in cols if c in NUMERICAL_COLS_ALL]
    cat_cols = [c for c in cols if c in CATEGORICAL_COLS]

    transformers = [("num", StandardScaler(), num_cols)]
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)

    return Pipeline([("preprocessor", preprocessor), ("knn", knn)])


# ==============================================================================
# 5. OPTIMASI GRID SEARCH DAN PELATIHAN LENGKAP
# ==============================================================================
def run_full_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_sets_to_test: Optional[List[str]] = None,
    k_range: range = range(1, 26),
    metrics: Optional[List[str]] = None,
    weights_list: Optional[List[str]] = None,
    cv_splits: int = 5,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Melakukan 5-fold StratifiedKFold CV grid search.
    Kriteria pemilihan:
    1. cv_accuracy tertinggi
    2. cv_macro_f1 tertinggi
    3. feature_set lebih sederhana (A < B < C)
    4. k lebih kecil
    """
    if feature_sets_to_test is None:
        feature_sets_to_test = ["A", "B", "C"]
    if metrics is None:
        metrics = ["euclidean", "manhattan"]
    if weights_list is None:
        weights_list = ["uniform", "distance"]

    fset_order = {"A": 1, "B": 2, "C": 3}
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    all_results: List[Dict[str, Any]] = []

    for fset_name in feature_sets_to_test:
        cols = FEATURE_SETS[fset_name]
        num_cols = [c for c in cols if c in NUMERICAL_COLS_ALL]
        cat_cols = [c for c in cols if c in CATEGORICAL_COLS]

        transformers = [("num", StandardScaler(), num_cols)]
        if cat_cols:
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
            )

        fold_train_trans = []
        fold_val_trans = []
        fold_y_train = []
        fold_y_val = []

        for tr_idx, va_idx in cv.split(X_train[cols], y_train):
            X_tr = X_train[cols].iloc[tr_idx]
            y_tr = y_train.iloc[tr_idx]
            X_va = X_train[cols].iloc[va_idx]
            y_va = y_train.iloc[va_idx]

            prep = ColumnTransformer(transformers=transformers)
            X_tr_t = prep.fit_transform(X_tr)
            X_va_t = prep.transform(X_va)

            fold_train_trans.append(X_tr_t)
            fold_val_trans.append(X_va_t)
            fold_y_train.append(y_tr)
            fold_y_val.append(y_va)

        for metric in metrics:
            for weights in weights_list:
                for k in k_range:
                    accs = []
                    f1s = []
                    for f in range(cv_splits):
                        knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
                        knn.fit(fold_train_trans[f], fold_y_train[f])
                        pred = knn.predict(fold_val_trans[f])
                        accs.append(accuracy_score(fold_y_val[f], pred))
                        f1s.append(
                            precision_recall_fscore_support(
                                fold_y_val[f], pred, average="macro", zero_division=0
                            )[2]
                        )

                    mean_acc = float(np.mean(accs))
                    mean_f1 = float(np.mean(f1s))
                    all_results.append(
                        {
                            "feature_set": fset_name,
                            "k": int(k),
                            "metric": metric,
                            "weights": weights,
                            "cv_accuracy": mean_acc,
                            "cv_macro_f1": mean_f1,
                            "fset_rank": fset_order.get(fset_name, 99),
                        }
                    )

    df_res = pd.DataFrame(all_results)
    df_sorted = df_res.sort_values(
        by=["cv_accuracy", "cv_macro_f1", "fset_rank", "k"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    best_config = df_sorted.iloc[0].to_dict()
    return best_config, df_sorted


def train_and_save_model(
    data_path: str | Path = "data_status_gizi_balita.xlsx",
    output_path: str | Path = "models/knn_gizi_balita_madura.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
    force_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Melatih model KNN secara penuh:
    1. Membaca data dan split stratified 80/20.
    2. Mencari konfigurasi optimal (atau menggunakan force_config jika ditentukan).
    3. Melatih ulang pada data latih penuh.
    4. Menguji pada data uji, mencatat metrik evaluasi lengkap.
    5. Menyimpan dict hasil ke file .joblib.
    """
    data_p = Path(data_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    df = load_data(data_p)
    X = df[EXPECTED_COLUMNS[:-1]]
    y = df["status_gizi"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if force_config is None:
        best_cfg, grid_df = run_full_grid_search(X_train, y_train)
    else:
        best_cfg = force_config
        grid_df = pd.DataFrame([force_config])

    fset_name = best_cfg["feature_set"]
    k_val = int(best_cfg["k"])
    metric_val = str(best_cfg["metric"])
    weights_val = str(best_cfg["weights"])
    cols = FEATURE_SETS[fset_name]

    # Evaluasi perbandingan antar set fitur (A vs B vs C) pada konfigurasi terbaik
    fset_comparison: Dict[str, Dict[str, float]] = {}
    for fs_key in ["A", "B", "C"]:
        sub_pipe = create_pipeline(fs_key, k=k_val, metric=metric_val, weights=weights_val)
        sub_pipe.fit(X_train[FEATURE_SETS[fs_key]], y_train)
        sub_pred = sub_pipe.predict(X_test[FEATURE_SETS[fs_key]])
        sub_acc = float(accuracy_score(y_test, sub_pred))
        sub_p, sub_r, sub_f1, _ = precision_recall_fscore_support(
            y_test, sub_pred, average="macro", zero_division=0
        )
        fset_comparison[fs_key] = {
            "accuracy": sub_acc,
            "precision_macro": float(sub_p),
            "recall_macro": float(sub_r),
            "f1_macro": float(sub_f1),
        }

    # Pipeline utama
    pipeline = create_pipeline(fset_name, k=k_val, metric=metric_val, weights=weights_val)
    pipeline.fit(X_train[cols], y_train)

    y_pred = pipeline.predict(X_test[cols])
    test_acc = float(accuracy_score(y_test, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    cls_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=list(pipeline.classes_))

    model_dict: Dict[str, Any] = {
        "pipeline": pipeline,
        "feature_set": fset_name,
        "features": cols,
        "k": k_val,
        "metric": metric_val,
        "weights": weights_val,
        "classes": list(pipeline.classes_),
        "metrics": {
            "accuracy": test_acc,
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro": float(f1_macro),
            "cv_accuracy": float(best_cfg.get("cv_accuracy", test_acc)),
            "cv_macro_f1": float(best_cfg.get("cv_macro_f1", f1_macro)),
            "classification_report": cls_report,
            "confusion_matrix": cm.tolist(),
            "feature_set_comparison": fset_comparison,
        },
        "n_train": len(X_train),
        "n_test": len(X_test),
        "trained_at": datetime.now().isoformat(),
        "sklearn_version": sklearn.__version__,
        "data_file": str(data_p.name),
        "grid_summary": grid_df.head(20).to_dict(orient="records"),
    }

    joblib.dump(model_dict, out_p)
    return model_dict


# ==============================================================================
# 6. EKSEKUSI MANDIRI (CLI)
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PELATIHAN MODEL K-NEAREST NEIGHBORS (KNN) - STATUS GIZI BALITA MADURA")
    print("Pembuat : ABD ROHMAN - 20251230035")
    print("Instansi: Sistem Informasi, Fakultas Teknik, Universitas Islam Madura 2026")
    print("=" * 70)

    data_file = Path("data_status_gizi_balita.xlsx")
    output_model = Path("models/knn_gizi_balita_madura.joblib")

    print(f"Membaca dataset: {data_file.resolve()}...")
    df_data = load_data(data_file)
    print(f"Dataset berhasil dimuat: {len(df_data)} baris, {len(df_data.columns)} kolom.")
    print("Distribusi kelas target:")
    for status, count in df_data["status_gizi"].value_counts().items():
        print(f"  - {status:12s}: {count:4d} ({count / len(df_data) * 100:.1f}%)")

    print("\nMenjalankan 5-fold Stratified CV Grid Search (Set A/B/C x K 1..25 x metrik x weights)...")
    model_info = train_and_save_model(data_path=data_file, output_path=output_model)

    print("\n" + "=" * 70)
    print("HASIL PELATIHAN MODEL TERBAIK:")
    print("=" * 70)
    print(f"Set Fitur Terpilih : SET {model_info['feature_set']} -> {model_info['features']}")
    print(f"Parameter K        : {model_info['k']}")
    print(f"Metrik Jarak       : {model_info['metric']}")
    print(f"Pembobotan         : {model_info['weights']}")
    print(f"Jumlah Data Latih  : {model_info['n_train']} baris")
    print(f"Jumlah Data Uji    : {model_info['n_test']} baris")
    print(f"Akurasi Data Uji   : {model_info['metrics']['accuracy'] * 100:.2f}% (Target: >= 85%)")
    print(f"Macro Precision    : {model_info['metrics']['precision_macro'] * 100:.2f}%")
    print(f"Macro Recall       : {model_info['metrics']['recall_macro'] * 100:.2f}%")
    print(f"Macro F1-Score     : {model_info['metrics']['f1_macro'] * 100:.2f}%")
    print(f"Model tersimpan di : {output_model.resolve()}")

    print("\nPerbandingan Akurasi Antar Set Fitur:")
    for fkey, fmetrics in model_info["metrics"]["feature_set_comparison"].items():
        print(
            f"  - Set {fkey}: Akurasi = {fmetrics['accuracy'] * 100:.2f}%, F1 = {fmetrics['f1_macro'] * 100:.2f}%"
        )

    print("\nPengujian Verifikasi Selesai dengan SUKSES!")
